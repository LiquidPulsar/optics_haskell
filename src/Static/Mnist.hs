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
  let rss = case [ w | l <- lines status
                     , "VmRSS" `isPrefixOf` l
                     , w <- words l, all isDigit w ] of
              (r:_) -> r
              []    -> "unknown"
  putStrLn $ "GHC live: " <> show (liveBytes `div` 1024) <> " KB"
           <> "  RSS: " <> rss <> " kB"

-- ─── Architecture ─────────────────────────────────────────────────────────────
-- [batch,  1, 28, 28]
--   conv1 (3×3, no bias) -> relu -> maxpool (2×2) -> [batch,  3, 13, 13]
--   conv2 (4×4, no bias) -> relu -> maxpool (2×2) -> [batch,  5,  5,  5]
--   flatten                                       -> [batch, 125]
--   dense (MMP 10 125)   -> sigmoid               -> [batch,  10]

type BatchSize = 32
-- 10x too much seemingly
type NumTrain  = 6000--0
type NumTest   = 1000--0

-- Conv layers carry only a kernel (no bias — kept separate to match HMatrix style)
type Conv1K dev dt = Tensor dev dt [3, 1, 3, 3]
type Conv2K dev dt = Tensor dev dt [5, 3, 4, 4]
type DenseP dev dt = MMP dev dt 10 125
type MnistP dev dt = (Conv1K dev dt, (Conv2K dev dt, DenseP dev dt))

type SaneMnist dev dt =
  ( CanMMLens dev dt
  , CanAddLens dev dt
  , T.StandardFloatingPointDTypeValidation dev dt
  , T.ComparisonDTypeIsValid dev dt
  , T.MeanDTypeValidation dev dt
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
  Tensor dev dt (n : shape) ->
  [Tensor dev dt (b : shape)]
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
    (Inp (t [b, 1, 28, 28]), MnistP dev dt)
    ()
    (Out (t [b, 10]))
mnistModel = argToPara -- 28
  .#. withGradDesc convLens . relu . maxPool @'(2, 2) @'(2, 2) @'(0, 0) -- 13
  .#. withGradDesc convLens . relu . maxPool @'(2, 2) @'(2, 2) @'(0, 0) -- 5
  .#. rightLens (flatten @b @[5, 5, 5]) . matMulLens

mnistModelLoss ::
  forall b dev dt t.
  ( t ~ Tensor dev dt
  , KnownNat b
  , T.AllDimsPositive '[b]
  , SaneMnist dev dt
  ) =>
  ParaLens'
    ((Inp (t [b, 1, 28, 28]), MnistP dev dt), Tgt (t [b, 10]))
    ()
    (Out (t '[]))
mnistModelLoss = mnistModel .#. softMaxCELoss

mnistModel' ::
  forall b dev dt t.
  ( t ~ Tensor dev dt
  , KnownNat b
  , T.AllDimsPositive '[b]
  , SaneMnist dev dt
  ) =>
  ParaLens'
    ((Inp (t [b, 1, 28, 28]), MnistP dev dt), Tgt (t [b, 10]))
    ()
    ()
mnistModel' = mnistModel .#. softMaxCELoss . lrSmooth 1e-4

-- ─── Initialisation ───────────────────────────────────────────────────────────

mnistInitParams ::
  forall dev dt.
  ( T.KnownDType dt
  , T.RandDTypeIsValid dev dt
  , T.KnownDevice dev
  ) =>
  IO (MnistP dev dt)
mnistInitParams = do
  -- He init: std = sqrt(2 / fan_in), prevents sigmoid saturation in dense layer
  -- Conv1 [3,1,3,3] fan_in = 1*3*3 = 9
  -- Conv2 [5,3,4,4] fan_in = 3*4*4 = 48
  -- Dense [10,125]  fan_in = 125
  let sc x = T.mulScalar (x :: Float)
  c1 <- sc (sqrt (2/9))   <$> T.randn
  c2 <- sc (sqrt (2/48))  <$> T.randn
  w  <- sc (sqrt (2/125)) <$> T.randn
  let b = T.zeros
  pure (c1, (c2, (w, b)))

-- ─── Data loading ─────────────────────────────────────────────────────────────

loadMnist ::
  forall n dev dt.
  ( KnownNat n
  , T.KnownDevice dev
  , T.KnownDType dt
  ) =>
  FilePath ->
  FilePath ->
  IO (Tensor dev dt [n, 1, 28, 28], Tensor dev dt [n, 10])
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
  Tensor dev dt [n, 1, 28, 28] ->
  Tensor dev dt [n, 10] ->
  [( Tensor dev dt [BatchSize, 1, 28, 28]
   , Tensor dev dt [BatchSize, 10] )]
mnistTargets imgs lbls =
  zip (batchesOf @BatchSize imgs) (batchesOf @BatchSize lbls)

-- ─── Epoch & accuracy ─────────────────────────────────────────────────────────

mnistEpoch ::
  forall dev dt.
  SaneMnist dev dt =>
  [( Tensor dev dt [BatchSize, 1, 28, 28]
   , Tensor dev dt [BatchSize, 10] )] ->
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
  t [b, 1, 28, 28] ->
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
  [( Tensor dev dt [BatchSize, 1, 28, 28]
   , Tensor dev dt [BatchSize, 10] )] ->
  Double
mnistAccuracy p targets = fromIntegral correct / fromIntegral total
  where
    results = flip concatMap targets $ \(imgs, lbls) ->
      let predicted = mnistPredict p imgs
          actual    = U.asValue . toDynamic $ T.argmax @1 @T.DropDim lbls :: [Int]
      in zipWith (==) predicted actual
    correct = length (filter id results)
    total   = length results

-- ─── Diagnostics ──────────────────────────────────────────────────────────────

mnistDiagnose ::
  forall dev dt.
  ( SaneMnist dev dt
  , T.StandardDTypeValidation dev dt
  ) =>
  MnistP dev dt ->
  [( Tensor dev dt [BatchSize, 1, 28, 28]
   , Tensor dev dt [BatchSize, 10] )] ->
  IO ()
mnistDiagnose p testT = do
  -- Prediction class histogram
  let preds  = concatMap (\(imgs, _) -> mnistPredict p imgs) testT
      counts = map (\c -> (c, length (filter (== c) preds))) [0 .. 9]
  putStrLn "Prediction histogram:"
  forM_ counts $ \(c, n) ->
    putStrLn $ "  class " <> show c <> ": " <> show n

  -- Parameter stats: mean, std (NaN shows up as NaN here)
  let (c1, (c2, (w, b))) = p
      stat lbl t =
        let d = toDynamic t
            m = U.asValue (U.mean d) :: Float
            s = U.asValue (U.std  d) :: Float
        in putStrLn $ "  " <> lbl
             <> "  mean=" <> show m
             <> "  std="  <> show s
             <> if isNaN m || isNaN s then "  [NaN!]" else ""
  putStrLn "Parameter stats:"
  stat "conv1  " c1
  stat "conv2  " c2
  stat "dense_w" w
  stat "dense_b" b

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

  -- let paramL1Diff (c1,  (c2,  (w,  b)))
  --                 (c1', (c2', (w', b'))) =
  --       let diff t t' = U.asValue . U.sumAll . U.abs
  --                         $ U.sub (toDynamic t) (toDynamic t') :: Float
  --       in [ ("conv1",   diff c1 c1')
  --          , ("conv2",   diff c2 c2')
  --          , ("dense_w", diff w  w' )
  --          , ("dense_b", diff b  b' ) ]

  let epochs = iterate (mnistEpoch trainT) p0
  forM_ (zip [0 ..] epochs) $ \(e, p) -> do
  -- forM_ (zip [0 ..] (zip epochs (tail epochs))) $ \(e, (p, p')) -> do
    performMajorGC
    let acc   = mnistAccuracy p testT
        -- diffs = paramL1Diff p p'
    t1 <- getCurrentTime
    putStrLn $ unwords
      [ "epoch",    show (e  :: Int)
      , "accuracy", show acc
      , "time",     show (diffUTCTime t1 t0)
      ]
    -- forM_ diffs $ \(name, d) -> putStrLn $ "  " <> name <> " Δ=" <> show d