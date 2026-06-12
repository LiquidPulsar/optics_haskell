{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE NoStarIsType #-}

module Static.Autoencoder where

import Core
import Data.Foldable (forM_)
import Data.Time (getCurrentTime, diffUTCTime)
import GHC.TypeNats
import Models (runFullModel, runOne, trainMany)
import Static.Layers (CanMMLens, MMP, matMulLens, relu, sigmoid)
import Static.Loss (TrivialFacts, lossSmooth)
import Static.Mnist (NumTrain, batchesOf, loadMnist)
import Static.Optim (lrSmooth)
import System.IO (BufferMode (..), hSetBuffering, stdout)
import qualified Torch as U
import Torch.Typed (Tensor (UnsafeMkTensor), natValI, toDynamic)
import qualified Torch.Typed as T

type BatchSize = 32
type InputDim  = 784
type HiddenDim = 128
type LatentDim = 32


-- encoder: [b,784] -> Dense(784→128)+ReLU -> Dense(128→32)
type EncoderP dev dt = (MMP dev dt HiddenDim InputDim, MMP dev dt LatentDim HiddenDim)

-- decoder: [b,32] -> Dense(32→128)+ReLU -> Dense(128→784)+Sigmoid
type DecoderP dev dt = (MMP dev dt HiddenDim LatentDim, MMP dev dt InputDim HiddenDim)

type AEP dev dt = (EncoderP dev dt, DecoderP dev dt)


type SaneAE dev dt =
  ( CanMMLens dev dt
  , T.StandardFloatingPointDTypeValidation dev dt
  , T.ComparisonDTypeIsValid dev dt
  , T.MeanDTypeValidation dev dt
  , T.KnownDevice dev
  , T.KnownDType dt
  )


encoderCore ::
  forall b dev dt.
  (SaneAE dev dt, KnownNat b) =>
  ParaLens'
    (EncoderP dev dt)
    (T.Tensor dev dt [b, InputDim])
    (T.Tensor dev dt [b, LatentDim])
encoderCore = matMulLens . relu .#. matMulLens
{-# INLINE encoderCore #-}

decoderCore ::
  forall b dev dt.
  (SaneAE dev dt, KnownNat b) =>
  ParaLens'
    (DecoderP dev dt)
    (T.Tensor dev dt [b, LatentDim])
    (T.Tensor dev dt [b, InputDim])
decoderCore = matMulLens . relu .#. matMulLens . sigmoid
{-# INLINE decoderCore #-}


encoderModel ::
  forall b dev dt.
  (SaneAE dev dt, KnownNat b) =>
  ParaLens'
    (T.Tensor dev dt [b, InputDim], EncoderP dev dt)
    ()
    (T.Tensor dev dt [b, LatentDim])
encoderModel = argToPara .#. encoderCore
{-# INLINE encoderModel #-}

decoderModel ::
  forall b dev dt.
  (SaneAE dev dt, KnownNat b) =>
  ParaLens'
    (T.Tensor dev dt [b, LatentDim], DecoderP dev dt)
    ()
    (T.Tensor dev dt [b, InputDim])
decoderModel = argToPara .#. decoderCore
{-# INLINE decoderModel #-}

autoencoderModel ::
  forall b dev dt.
  (SaneAE dev dt, KnownNat b) =>
  ParaLens'
    (T.Tensor dev dt [b, InputDim], AEP dev dt)
    ()
    (T.Tensor dev dt [b, InputDim])
autoencoderModel = argToPara .#. encoderCore .#. decoderCore
{-# INLINE autoencoderModel #-}

-- ─── Loss variants ────────────────────────────────────────────────────────────

autoencoderModelLoss ::
  forall b dev dt.
  ( SaneAE dev dt
  , KnownNat b
  , TrivialFacts [b, InputDim]
  ) =>
  ParaLens'
    ( (T.Tensor dev dt [b, InputDim], AEP dev dt)
    , T.Tensor dev dt [b, InputDim] )
    ()
    (T.Tensor dev dt '[])
autoencoderModelLoss = autoencoderModel .#. lossSmooth
{-# INLINE autoencoderModelLoss #-}

autoencoderModel' ::
  forall b dev dt.
  ( SaneAE dev dt
  , KnownNat b
  , TrivialFacts [b, InputDim]
  ) =>
  ParaLens'
    ( (T.Tensor dev dt [b, InputDim], AEP dev dt)
    , T.Tensor dev dt [b, InputDim] )
    ()
    ()
autoencoderModel' = autoencoderModel .#. lossSmooth . lrSmooth 1e-3
{-# INLINE autoencoderModel' #-}


natValF :: forall i . KnownNat i => Float
natValF = fromIntegral $ natValI @i

aeInitParams ::
  forall dev dt.
  ( T.KnownDType dt
  , T.RandDTypeIsValid dev dt
  , T.KnownDevice dev
  ) =>
  IO (AEP dev dt)
aeInitParams = do
  let sc x = T.mulScalar (x :: Float)
  w1 <- sc (sqrt (2 / natValF @InputDim)) <$> T.randn ; let b1 = T.zeros
  w2 <- sc (sqrt (2 / natValF @HiddenDim)) <$> T.randn ; let b2 = T.zeros
  w3 <- sc (sqrt (2 / natValF @LatentDim))  <$> T.randn ; let b3 = T.zeros
  w4 <- sc (sqrt (2 / natValF @HiddenDim)) <$> T.randn ; let b4 = T.zeros
  pure (((w1, b1), (w2, b2)), ((w3, b3), (w4, b4)))

flattenImgs ::
  forall n dev dt.
  KnownNat n =>
  T.Tensor dev dt [n, 1, 28, 28] ->
  T.Tensor dev dt [n, InputDim]
flattenImgs t = UnsafeMkTensor $ U.reshape [natValI @n, 784] (toDynamic t)

aeTargets ::
  forall dev dt.
  T.Tensor dev dt [NumTrain, InputDim] ->
  [( T.Tensor dev dt [BatchSize, InputDim]
   , T.Tensor dev dt [BatchSize, InputDim] )]
aeTargets flat = let bs = batchesOf @BatchSize flat in zip bs bs


aeEpoch ::
  forall dev dt.
  ( SaneAE dev dt
  , TrivialFacts [BatchSize, InputDim]
  ) =>
  [( T.Tensor dev dt [BatchSize, InputDim]
   , T.Tensor dev dt [BatchSize, InputDim] )] ->
  AEP dev dt ->
  AEP dev dt
aeEpoch = flip (trainMany (autoencoderModel' @BatchSize))

aeTrain :: IO ()
aeTrain = do
  hSetBuffering stdout LineBuffering
  (trainImgs, _) <-
    loadMnist @NumTrain
      "data/train-images-idx3-ubyte"
      "data/train-labels-idx1-ubyte"
  let flat   = flattenImgs trainImgs
      trainT = aeTargets flat
  p0 <- aeInitParams @'(T.CPU, 0) @T.Float
  t0 <- getCurrentTime
  let epochs = iterate (aeEpoch trainT) p0
  forM_ (zip [0 ..] epochs) $ \(e, p) -> do
    let loss = aeReconstructionLoss p trainT
    t1 <- getCurrentTime
    putStrLn $ unwords
      [ "epoch",    show (e :: Int)
      , "loss",     show loss
      , "time",     show (diffUTCTime t1 t0) ]

encode ::
  forall b dev dt.
  (SaneAE dev dt, KnownNat b) =>
  AEP dev dt ->
  T.Tensor dev dt [b, InputDim] ->
  T.Tensor dev dt [b, LatentDim]
encode (ep, _) x = runFullModel encoderModel (x, ep)

decode ::
  forall b dev dt.
  (SaneAE dev dt, KnownNat b) =>
  AEP dev dt ->
  T.Tensor dev dt [b, LatentDim] ->
  T.Tensor dev dt [b, InputDim]
decode (_, dp) z = runFullModel decoderModel (z, dp)

reconstruct ::
  forall b dev dt.
  (SaneAE dev dt, KnownNat b) =>
  AEP dev dt ->
  T.Tensor dev dt [b, InputDim] ->
  T.Tensor dev dt [b, InputDim]
reconstruct p = decode p . encode p

aeReconstructionLoss ::
  forall dev dt.
  ( SaneAE dev dt
  , TrivialFacts [BatchSize, InputDim]
  ) =>
  AEP dev dt ->
  [( T.Tensor dev dt [BatchSize, InputDim]
   , T.Tensor dev dt [BatchSize, InputDim] )] ->
  Float
aeReconstructionLoss p targets =
  U.asValue . toDynamic . T.divScalar (fromIntegral n :: Float) $
    sum (map (\(x, _) -> runOne autoencoderModelLoss p (x, x)) targets)
  where
    n = length targets
