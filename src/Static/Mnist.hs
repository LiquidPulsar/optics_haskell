{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE ConstraintKinds #-}

module Static.Mnist where

import Data.Foldable (forM_)
import Data.IDX
import Data.Maybe (fromJust)
import Data.Time (getCurrentTime, diffUTCTime)
import GHC.TypeNats
import System.IO (hSetBuffering, stdout, BufferMode (..))
import qualified Data.Vector.Unboxed as VU
import qualified Torch as U
import qualified Torch.Typed as T
import Torch.Typed (Tensor, Tensor(UnsafeMkTensor), toDynamic, natValI)
import qualified Torch.Functional.Internal as I

import Core
import Types hiding (MMP)
import Static.Layers
import Static.Loss
import Static.Optim
import Models
import Stack (RandInit(randInit))

import System.Mem (performMajorGC)
import GHC.Stats (getRTSStats, gcdetails_live_bytes, gc)
import Data.List
import Data.Char

memStats :: IO ()
memStats = do
  performMajorGC   -- force GC so we see actual live bytes, not garbage
  rts <- getRTSStats
  -- GHC heap
  let liveBytes = gcdetails_live_bytes (gc rts)
  -- ATen tensor memory (not visible to GHC) — read from /proc
  status <- readFile "/proc/self/status"
  let rss = head [ w | l <- lines status
                     , "VmRSS" `isPrefixOf` l
                     , w <- words l, all isDigit w ]
  putStrLn $ "GHC live: " <> show (liveBytes `div` 1024) <> " KB"
           <> "  RSS: " <> rss <> " kB"

-- ─── Architecture ─────────────────────────────────────────────────────────────
-- [batch,  1, 28, 28]
--   conv1 (3×3, no bias) → relu → maxpool (2×2) → [batch,  3, 13, 13]
--   conv2 (4×4, no bias) → relu → maxpool (2×2) → [batch,  5,  5,  5]
--   flatten                                      → [batch, 125]
--   dense (MMP 10 125)   → sigmoid               → [batch,  10]

type BatchSize = 32
type NumTrain  = 6000--0
type NumTest   = 1000--0

-- Conv layers carry only a kernel (no bias — kept separate to match HMatrix style)
type Conv1K dev dt = Tensor dev dt '[3, 1, 3, 3]
type Conv2K dev dt = Tensor dev dt '[5, 3, 4, 4]
type DenseP dev dt = MMP dev dt 10 125
type MnistP dev dt = (Conv1K dev dt, (Conv2K dev dt, DenseP dev dt))

type SaneMnist dev dt =
  ( CanMMLens dev dt
  , CanAddLens dev dt
  , T.StandardFloatingPointDTypeValidation dev dt
  , T.ComparisonDTypeIsValid dev dt
  , T.KnownDevice dev
  , T.KnownDType dt
  )

-- ─── General batch slicer ──────────────────────────────────────────────────────

-- Slices along dim 0; works for any trailing shape, so handles both
-- image tensors [n, 1, 28, 28] and label tensors [n, 10].
batchesOf ::
  forall b n shape dev dt.
  ( KnownNat b
  , KnownNat n
  ) =>
  Tensor dev dt (n ': shape) ->
  [Tensor dev dt (b ': shape)]
batchesOf t =
  [ UnsafeMkTensor $ I.narrow_tlll (toDynamic t) 0 (i * b') b'
  | i <- [0 .. (n' `div` b') - 1] ]
  where
    b' = natValI @b
    n' = natValI @n

-- ─── Model ────────────────────────────────────────────────────────────────────

mnistModel ::
  forall b dev dt t.
  ( t ~ Tensor dev dt
  , KnownNat b
  , SaneMnist dev dt
  ) =>
  ParaLens'
    (Inp (t '[b, 1, 28, 28]), MnistP dev dt)
    ()
    (Out (t '[b, 10]))
mnistModel = argToPara -- 28
  .#. convLens . relu . maxPool @'(2, 2) @'(2, 2) @'(0, 0) -- 13
  .#. convLens . relu . maxPool @'(2, 2) @'(2, 2) @'(0, 0) -- 5
  .#. rightLens (flatten @b @'[5, 5, 5]) . matMulLens . sigmoid

mnistModelLoss ::
  forall b dev dt t.
  ( t ~ Tensor dev dt
  , KnownNat b
  , SaneMnist dev dt
  ) =>
  ParaLens'
    ((Inp (t '[b, 1, 28, 28]), MnistP dev dt), Tgt (t '[b, 10]))
    ()
    (Out (t '[]))
mnistModelLoss = mnistModel .#. lossSmooth

mnistModel' ::
  forall b dev dt t.
  ( t ~ Tensor dev dt
  , KnownNat b
  , SaneMnist dev dt
  ) =>
  ParaLens'
    ((Inp (t '[b, 1, 28, 28]), MnistP dev dt), Tgt (t '[b, 10]))
    ()
    ()
mnistModel' = mnistModel .#. lossSmooth . lrSmooth 0.01

-- ─── Initialisation ───────────────────────────────────────────────────────────

mnistInitParams ::
  forall dev dt.
  ( T.KnownDType dt
  , T.RandDTypeIsValid dev dt
  , T.KnownDevice dev
  ) =>
  IO (MnistP dev dt)
mnistInitParams = randInit
--   (,,) <$> T.randn                   -- Conv1K: [3, 1, 3, 3]
--        <*> T.randn                   -- Conv2K: [5, 3, 4, 4]
--        <*> liftA2 (,) T.randn T.randn  -- DenseP: (weight, bias)
--   <&> \(c1, c2, d) -> (c1, (c2, d))

-- ─── Data loading ─────────────────────────────────────────────────────────────

loadMnist ::
  forall n dev dt.
  ( KnownNat n
  , T.KnownDevice dev
  , T.KnownDType dt
  ) =>
  FilePath ->
  FilePath ->
  IO (Tensor dev dt '[n, 1, 28, 28], Tensor dev dt '[n, 10])
loadMnist imgPath lblPath = do
  Just imgs <- decodeIDXFile imgPath
  Just lbls <- decodeIDXLabelsFile lblPath
  let pairs = take n $ fromJust $ labeledDoubleData lbls imgs
      n    = natValI @n
      convert =
          UnsafeMkTensor
        . U.toDevice (T.deviceVal @dev)
        . U.toType   (T.dtypeVal  @dt)
      imgT = convert
           . U.reshape [n, 1, 28, 28]
           . U.asTensor @[Double] -- annotation solves the next line too
           $ concatMap (map realToFrac . VU.toList . snd) pairs
      lblT = convert
           . U.oneHot 10
           . U.asTensor
           $ map fst pairs
  pure (imgT, lblT)

-- ─── Training targets ─────────────────────────────────────────────────────────

mnistTargets ::
  KnownNat n =>
  Tensor dev dt '[n, 1, 28, 28] ->
  Tensor dev dt '[n, 10] ->
  [( Tensor dev dt '[BatchSize, 1, 28, 28]
   , Tensor dev dt '[BatchSize, 10] )]
mnistTargets imgs lbls =
  zip (batchesOf @BatchSize imgs) (batchesOf @BatchSize lbls)

-- ─── Epoch & accuracy ─────────────────────────────────────────────────────────

mnistEpoch ::
  forall dev dt.
  SaneMnist dev dt =>
  [( Tensor dev dt '[BatchSize, 1, 28, 28]
   , Tensor dev dt '[BatchSize, 10] )] ->
  MnistP dev dt ->
  MnistP dev dt
mnistEpoch = flip (trainMany mnistModel')

mnistPredict ::
  forall b dev dt t.
  ( t ~ Tensor dev dt
  , KnownNat b
  , SaneMnist dev dt
  , T.StandardDTypeValidation dev dt
  ) =>
  MnistP dev dt ->
  t '[b, 1, 28, 28] ->
  [Int]
mnistPredict p x =
  U.asValue . toDynamic $
    T.argmax @1 @T.DropDim (runFullModel mnistModel (x, p))

mnistAccuracy ::
  forall dev dt.
  ( SaneMnist dev dt
  , T.StandardDTypeValidation dev dt
  ) =>
  MnistP dev dt ->
  [( Tensor dev dt '[BatchSize, 1, 28, 28]
   , Tensor dev dt '[BatchSize, 10] )] ->
  Double
mnistAccuracy p targets = fromIntegral correct / fromIntegral total
  where
    results = flip concatMap targets $ \(imgs, lbls) ->
      let predicted = mnistPredict p imgs
          actual    = U.asValue . toDynamic $ T.argmax @1 @T.DropDim lbls :: [Int]
      in zipWith (==) predicted actual
    correct = length (filter id results)
    total   = length results

-- ─── Entry point ──────────────────────────────────────────────────────────────

mnistTrain :: IO ()
mnistTrain = do
  hSetBuffering stdout LineBuffering
  (trainImgs, trainLbls) <-
    loadMnist @NumTrain
      "data/train-images-idx3-ubyte"
      "data/train-labels-idx1-ubyte"
  (testImgs, testLbls) <-
    loadMnist @NumTest
      "data/t10k-images-idx3-ubyte"
      "data/t10k-labels-idx1-ubyte"

  let trainT = mnistTargets trainImgs trainLbls
      testT  = mnistTargets testImgs  testLbls

  p0 <- mnistInitParams @'(T.CPU, 0) @T.Float
  t0 <- getCurrentTime

  let epochs = iterate (mnistEpoch trainT) p0
  forM_ (zip [0 ..] epochs) $ \(e, p) -> do
    performMajorGC
    -- memStats
    let acc = mnistAccuracy p testT
    t1 <- getCurrentTime
    putStrLn $ unwords
      [ "epoch",    show (e  :: Int)
      , "accuracy", show acc
      , "time",     show (diffUTCTime t1 t0)
      ]