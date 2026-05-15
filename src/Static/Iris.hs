{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}

module Static.Iris where

import Control.Arrow
-- import Types
import Core
import Data.Function
import Data.Functor
import qualified Data.List as L
import Data.Maybe
import GHC.TypeLits
import IrisData
import Models (runFullModel, runOne, trainMany)
import Stack
import Static.Layers (CanMMLens, MMP, matMulLens, sigmoid, transp)
import Static.Loss
import Static.Optim
import Torch (Tensor, TensorLike (asTensor), asValue, oneHot, toDevice, toType)
import Torch.Functional.Internal (narrow_tlll)
import qualified Torch.Typed as T
import Types (Inp, Out, Tgt)

type BatchSize = 8

irisToVec :: Iris -> [Double]
irisToVec =
  flip
    map
    [ sepalLength,
      sepalWidth,
      petalLength,
      petalWidth
    ]
    . (&)

type NumIris = 150

convert :: forall dv dt shape. (T.KnownDevice dv, T.KnownDType dt) => Tensor -> T.Tensor dv dt shape
convert =
  T.UnsafeMkTensor
    . toDevice (T.deviceVal @dv)
    . toType (T.dtypeVal @dt)
{-# INLINE convert #-} -- already inlines but let's be safe

irisToTensor ::
  ( T.KnownDType dt,
    T.KnownDevice dv
  ) =>
  [Iris] ->
  ( T.Tensor dv dt '[NumIris, 4],
    T.Tensor dv dt '[NumIris, 3]
  )
irisToTensor = feats &&& classes
  where
    feats = convert . asTensor . map irisToVec
    classes = convert . oneHot 3 . asTensor . map (fromEnum . irisClass)

sliceBatch ::
  forall batchSize features n dv dt.
  ( KnownNat batchSize,
    KnownNat features
  ) =>
  Int -> -- runtime offset
  T.Tensor dv dt '[n, features] ->
  T.Tensor dv dt '[batchSize, features]
sliceBatch offset t = T.UnsafeMkTensor $ narrow_tlll (T.toDynamic t) 0 offset b
  where
    b = T.natValI @batchSize

batches :: forall n batch dv dt features. (T.All KnownNat [batch, features, n]) => T.Tensor dv dt '[n, features] -> [T.Tensor dv dt '[batch, features]]
batches dataset = [sliceBatch @batch (i * b) dataset | i <- [0 .. (n `div` b) - 1]]
  where
    b = T.natValI @batch
    n = T.natValI @n

irisTargets :: (T.KnownDevice dv, T.KnownDType dt) => [(T.Tensor dv dt '[BatchSize, 4], T.Tensor dv dt '[BatchSize, 3])]
irisTargets = zip (batches inps) (batches tgts)
  where
    (inps, tgts) = irisToTensor iris

type InnerLayer dv dt = MMP dv dt 4 4

type IParams n dv dt = (StackedN n (InnerLayer dv dt), MMP dv dt 3 4)

nMuls :: forall stack n dv dt b. (SaneDT dv dt, T.KnownDevice dv, CanStack stack) => ParaLens' (StackedN stack (MMP dv dt n n)) (T.Tensor dv dt '[b, n]) (T.Tensor dv dt '[b, n])
nMuls = stackN @stack (matMulLens . sigmoid)
{-# INLINE nMuls #-}

type SaneDT dv dt =
  ( T.KnownDType dt,
    T.StandardFloatingPointDTypeValidation dv dt,
    CanMMLens dv dt
  )

irisModel ::
  forall n dv dt b.
  ( CanStack n,
    SaneDT dv dt,
    T.KnownDevice dv
  ) =>
  ParaLens' (Inp (T.Tensor dv dt '[b, 4]), IParams n dv dt) () (Out (T.Tensor dv dt '[b, 3]))
irisModel = argToPara .#. nMuls @n .#. matMulLens . sigmoid
{-# INLINE irisModel #-}

test :: forall dv dt b. (SaneDT dv dt, T.KnownDevice dv) => (Inp (T.Tensor dv dt '[b, 4]), IParams 1 dv dt) -> Out (T.Tensor dv dt '[b, 3])
test = runFullModel $ irisModel @1

handRolledTest :: (SaneDT dv dt, T.KnownDevice dv) => (Inp (T.Tensor dv dt '[b, 4]), IParams 1 dv dt) -> Out (T.Tensor dv dt '[b, 3])
handRolledTest (i, (mb, mb')) = layer mb' . layer mb $ i
  where
    layer (m, b) = T.sigmoid . T.add b . (`T.matmul` transp m)
    {-# INLINE layer #-}

irisModelLoss ::
  forall n dv dt b.
  ( CanStack n,
    SaneDT dv dt,
    T.KnownDevice dv
  ) =>
  ParaLens' ((Inp (T.Tensor dv dt '[b, 4]), IParams n dv dt), Tgt (T.Tensor dv dt '[b, 3])) () (Out (T.Tensor dv dt '[]))
irisModelLoss = irisModel @n .#. lossSmooth
{-# INLINE irisModelLoss #-}

irisModel' ::
  forall n dv dt b.
  ( CanStack n,
    SaneDT dv dt,
    T.KnownDevice dv
  ) =>
  ParaLens' ((Inp (T.Tensor dv dt '[b, 4]), IParams n dv dt), Tgt (T.Tensor dv dt '[b, 3])) () ()
irisModel' = irisModel @n .#. lossSmooth . lrSmooth 0.01
{-# INLINE irisModel' #-}

irisEpoch :: forall n dv dt. (CanStack n, SaneDT dv dt, T.KnownDevice dv) => IParams n dv dt -> IParams n dv dt
irisEpoch mmp = trainMany (irisModel' @n) mmp irisTargets

-- initParams :: Int -> Int -> IO (Matrix Double, Vector Double)
-- initParams inputDim outputDim = do
--   w <- rand outputDim inputDim
--   b <- flatten <$> rand outputDim 1
--   let f x = scale 0.01 (x - 0.5)
--   return (f w, f b)

irisInitParams :: (RandStack n (InnerLayer dv dt), T.RandDTypeIsValid dv dt, T.KnownDType dt, T.KnownDevice dv) => IO (IParams n dv dt)
irisInitParams = randInit

irisBestParams :: forall n dv dt. (CanStack n, RandStack n (InnerLayer dv dt), SaneDT dv dt, T.KnownDType dt, T.RandDTypeIsValid dv dt, T.KnownDevice dv) => IO [IParams n dv dt]
irisBestParams = iterate (irisEpoch @n) <$> irisInitParams @n

irisGetEpoch :: forall n dv dt. (CanStack n, RandStack n (InnerLayer dv dt), SaneDT dv dt, T.KnownDType dt, T.RandDTypeIsValid dv dt, T.KnownDevice dv) => Int -> IO (IParams n dv dt)
irisGetEpoch i = irisBestParams @n <&> (!! i)

irisError :: forall n dv dt. (CanStack n, RandStack n (InnerLayer dv dt), SaneDT dv dt, T.KnownDevice dv) => IParams n dv dt -> T.Tensor dv dt '[]
irisError = sum . flip map irisTargets . runOne (irisModelLoss @n)

irisPredict :: forall n dv dt b. (CanStack n, KnownNat b, SaneDT dv dt, T.KnownDevice dv) => IParams n dv dt -> Inp (T.Tensor dv dt '[b, 4]) -> Tgt (T.Tensor dv dt '[b, 3])
irisPredict mmp = runFullModel (irisModel @n) . (,mmp)

labelToIndex :: (T.StandardDTypeValidation dv dt) => T.Tensor dv dt '[b, 3] -> [Int]
labelToIndex = asValue . T.toDynamic . T.argmax @1 @T.DropDim

labelToIrisClass :: (T.StandardDTypeValidation dv dt) => T.Tensor dv dt '[b, 3] -> [IrisClass]
labelToIrisClass = map toEnum . labelToIndex

irisPredict' :: forall n dv dt b. (CanStack n, KnownNat b, SaneDT dv dt, T.StandardDTypeValidation dv dt, T.KnownDevice dv) => IParams n dv dt -> Inp (T.Tensor dv dt '[b, 4]) -> [IrisClass]
irisPredict' mmp = labelToIrisClass . irisPredict @n mmp

irisAccuracy :: forall n dv dt. 
  ( CanStack n, SaneDT dv dt,
    T.StandardDTypeValidation dv dt,
    T.KnownDevice dv
  ) =>
  IParams n dv dt ->
  Double
irisAccuracy mmp = fromIntegral correct / fromIntegral total
  where
    -- run predictions and get ground truth labels for each batch
    batched = flip map irisTargets $ \(inp, tgt) ->
      let predicted = labelToIndex (irisPredict @n mmp inp)
          actual = labelToIndex tgt -- argmax of one-hot targets
       in zipWith (==) predicted actual

    results = concat batched
    correct = length (filter id results)
    total = length results

type Device = '(T.CPU, 0)

irisRes :: forall n. (CanStack n, RandStack n (InnerLayer Device T.Double)) => IO [(Double, Double)]
irisRes = map ((asValue . T.toDynamic . irisError @n) &&& irisAccuracy @n) <$> irisBestParams @n @Device @T.Double

epochsToAcc :: Double -> [Double] -> Int
epochsToAcc r = fst . fromJust . L.find ((>= r) . snd) . zip [0 ..]