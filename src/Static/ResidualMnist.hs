{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE NoStarIsType #-}

module Static.ResidualMnist where

import Data.Foldable (forM_)
import Data.Time (getCurrentTime, diffUTCTime)
import GHC.TypeNats
import System.IO (hSetBuffering, stdout, BufferMode(..))
import qualified Torch as U
import qualified Torch.Typed as T
import Torch.Typed (Tensor, toDynamic, natValI)

import Core
import Stack
import Models
import Static.Layers
import Static.Loss
import Static.Optim
import Static.Mnist ( loadMnist, mnistTargets
                    , BatchSize, NumTrain, NumTest, SaneMnist )
import System.Mem (performMajorGC)


type Hidden    = 128
type NumBlocks = 3

-- One residual block: two square dense layers
type ResBlockP dev dt = (MMP dev dt Hidden Hidden, MMP dev dt Hidden Hidden)

-- StackedN 3 p = (p, (p, p))
type ResMnistP dev dt =
  ( MMP dev dt Hidden 784
  , ( StackedN NumBlocks (ResBlockP dev dt)
    , MMP dev dt 10 Hidden ) )


type SaneRes b dev dt =
  ( SaneMnist dev dt
  , KnownNat b
  , Num (Tensor dev dt [b, Hidden])
  , T.Broadcast [b, Hidden] [b, Hidden] ~ [b, Hidden]
  , T.KnownShape [1, 28, 28]
  , KnownNat (T.Numel [1, 28, 28])
  , CanStack NumBlocks
  )


-- y = relu(W2(relu(W1 x))) + x
resBlock ::
  forall b dev dt.
  SaneRes b dev dt =>
  ParaLens' (ResBlockP dev dt) (Tensor dev dt [b, Hidden]) (Tensor dev dt [b, Hidden])
resBlock = skipPara (matMulLens . relu .#. matMulLens)

resMnistModel ::
  forall b dev dt.
  SaneRes b dev dt =>
  ParaLens'
    (Tensor dev dt [b, 1, 28, 28], ResMnistP dev dt)
    ()
    (Tensor dev dt [b, 10])
resMnistModel = argToPara
  .#. rightLens (flatten @b @'[1, 28, 28]) . matMulLens . relu
  .#. stackN @NumBlocks (resBlock @b)
  .#. matMulLens

resMnistModelLoss ::
  forall b dev dt.
  ( SaneRes b dev dt
  , T.AllDimsPositive '[b]
  ) =>
  ParaLens'
    ((Tensor dev dt [b, 1, 28, 28], ResMnistP dev dt), Tensor dev dt [b, 10])
    ()
    (Tensor dev dt '[])
resMnistModelLoss = resMnistModel .#. softMaxCELoss

resMnistModel' ::
  forall b dev dt.
  ( SaneRes b dev dt
  , T.AllDimsPositive '[b]
  ) =>
  ParaLens'
    ((Tensor dev dt [b, 1, 28, 28], ResMnistP dev dt), Tensor dev dt [b, 10])
    ()
    ()
resMnistModel' = resMnistModel .#. softMaxCELoss . lrSmooth 1e-4

-- ─── Initialisation ───────────────────────────────────────────────────────────


natValF :: forall i . KnownNat i => Float
natValF = fromIntegral $ natValI @i

-- He init: std = sqrt(2 / fan_in)
resMnistInitParams ::
  forall dev dt.
  ( T.KnownDType dt
  , T.RandDTypeIsValid dev dt
  , T.KnownDevice dev
  ) =>
  IO (ResMnistP dev dt)
resMnistInitParams = do
  let sc x = T.mulScalar (x :: Float)
  -- Input projection: fan_in = 784
  wIn <- sc (sqrt (2/784)) <$> T.randn ; let bIn = T.zeros
  -- Residual blocks: fan_in = Hidden = 128
  -- StackedN 3 p = (p, (p, p))
  let sch = sc (sqrt (2/natValF @Hidden)) <$> T.randn
  w1a <- sch; let b1a = T.zeros
  w1b <- sch; let b1b = T.zeros
  w2a <- sch; let b2a = T.zeros
  w2b <- sch; let b2b = T.zeros
  w3a <- sch; let b3a = T.zeros
  w3b <- sch; let b3b = T.zeros
  -- Output layer: fan_in = Hidden = 128
  wOut <- sc (sqrt (2/natValF @Hidden)) <$> T.randn ; let bOut = T.zeros
  pure ( (wIn, bIn)
       , ( ( ((w1a, b1a), (w1b, b1b))
           , ( ((w2a, b2a), (w2b, b2b))
             , ((w3a, b3a), (w3b, b3b)) ) )
         , (wOut, bOut) ) )

-- ─── Inference & accuracy ─────────────────────────────────────────────────────

resMnistPredict ::
  forall b dev dt.
  ( SaneRes b dev dt
  , T.StandardDTypeValidation dev dt
  ) =>
  ResMnistP dev dt ->
  Tensor dev dt [b, 1, 28, 28] ->
  [Int]
resMnistPredict p x =
  U.asValue . toDynamic $
    T.argmax @1 @T.DropDim (runFullModel resMnistModel (x, p))

resMnistAccuracy ::
  forall dev dt.
  ( SaneRes BatchSize dev dt
  , T.StandardDTypeValidation dev dt
  ) =>
  ResMnistP dev dt ->
  [( Tensor dev dt [BatchSize, 1, 28, 28]
   , Tensor dev dt [BatchSize, 10] )] ->
  Double
resMnistAccuracy p targets = fromIntegral correct / fromIntegral total
  where
    results = flip concatMap targets $ \(imgs, lbls) ->
      let predicted = resMnistPredict p imgs
          actual    = U.asValue . toDynamic $ T.argmax @1 @T.DropDim lbls :: [Int]
      in  zipWith (==) predicted actual
    correct = length (filter id results)
    total   = length results


resMnistEpoch ::
  forall dev dt.
  ( SaneRes BatchSize dev dt
  , T.AllDimsPositive '[BatchSize]
  ) =>
  [( Tensor dev dt [BatchSize, 1, 28, 28]
   , Tensor dev dt [BatchSize, 10] )] ->
  ResMnistP dev dt ->
  ResMnistP dev dt
resMnistEpoch = flip (trainMany (resMnistModel' @BatchSize))

resMnistTrain :: IO ()
resMnistTrain = do
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

  p0 <- resMnistInitParams @'(T.CPU, 0) @T.Float
  t0 <- getCurrentTime

  let epochs = iterate (resMnistEpoch trainT) p0
  forM_ (zip [0 ..] epochs) $ \(e, p) -> do
    performMajorGC
    let acc = resMnistAccuracy p testT
    t1 <- getCurrentTime
    putStrLn $ unwords
      [ "epoch",    show (e  :: Int)
      , "accuracy", show acc
      , "time",     show (diffUTCTime t1 t0)
      ]
