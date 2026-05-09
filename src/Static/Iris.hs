{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}

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

irisTargets :: (T.KnownDType dt, T.KnownDevice dv) => [(T.Tensor dv dt '[BatchSize, 4], T.Tensor dv dt '[BatchSize, 3])]
irisTargets = zip (batches inps) (batches tgts)
  where
    (inps, tgts) = irisToTensor iris

type IParams dv dt = MMP dv dt 3 4

type SaneDT dv dt =
  ( T.KnownDType dt,
    T.StandardFloatingPointDTypeValidation dv dt,
    CanMMLens dv dt
  )

irisModel ::
  ( SaneDT dv dt,
    T.KnownDevice dv
  ) =>
  ParaLens' (Inp (T.Tensor dv dt '[b, 4]), IParams dv dt) () (Out (T.Tensor dv dt '[b, 3]))
irisModel = argToPara .#. matMulLens . sigmoid
{-# INLINE irisModel #-}

test :: (SaneDT dv dt, T.KnownDevice dv) => (Inp (T.Tensor dv dt '[b, 4]), IParams dv dt) -> Out (T.Tensor dv dt '[b, 3])
test = runFullModel irisModel

handRolledTest :: (SaneDT dv dt, T.KnownDevice dv) => (Inp (T.Tensor dv dt '[b, 4]), IParams dv dt) -> Out (T.Tensor dv dt '[b, 3])
handRolledTest (i, (m, b)) =
  let res = T.matmul i (transp m)
      res' = T.add res b
      res'' = T.sigmoid res'
   in res''

irisModelLoss ::
  ( SaneDT dv dt,
    T.KnownDevice dv
  ) =>
  ParaLens' ((Inp (T.Tensor dv dt '[b, 4]), IParams dv dt), Tgt (T.Tensor dv dt '[b, 3])) () (Out (T.Tensor dv dt '[]))
irisModelLoss = irisModel .#. lossSmooth
{-# INLINE irisModelLoss #-}

irisModel' ::
  ( SaneDT dv dt,
    T.KnownDevice dv
  ) =>
  ParaLens' ((Inp (T.Tensor dv dt '[b, 4]), IParams dv dt), Tgt (T.Tensor dv dt '[b, 3])) () ()
irisModel' = irisModel .#. lossSmooth . lrSmooth 0.01
{-# INLINE irisModel' #-}

irisEpoch :: (SaneDT dv dt, T.KnownDevice dv) => IParams dv dt -> IParams dv dt
irisEpoch mmp = trainMany irisModel' mmp irisTargets

-- initParams :: Int -> Int -> IO (Matrix Double, Vector Double)
-- initParams inputDim outputDim = do
--   w <- rand outputDim inputDim
--   b <- flatten <$> rand outputDim 1
--   let f x = scale 0.01 (x - 0.5)
--   return (f w, f b)

irisInitParams :: (T.KnownDType dt, T.RandDTypeIsValid dv dt, T.KnownDevice dv) => IO (IParams dv dt)
irisInitParams = liftA2 (,) T.randn T.randn

irisBestParams :: (T.KnownDType dt, T.RandDTypeIsValid dv dt, SaneDT dv dt, T.KnownDevice dv) => IO [IParams dv dt]
irisBestParams = iterate irisEpoch <$> irisInitParams

irisGetEpoch :: (T.KnownDType dt, T.RandDTypeIsValid dv dt, SaneDT dv dt, T.KnownDevice dv) => Int -> IO (IParams dv dt)
irisGetEpoch i = irisBestParams <&> (!! i)

irisError :: (SaneDT dv dt, T.KnownDevice dv) => IParams dv dt -> T.Tensor dv dt '[]
irisError = sum . flip map irisTargets . runOne irisModelLoss

irisPredict :: (KnownNat b, SaneDT dv dt, T.KnownDevice dv) => IParams dv dt -> Inp (T.Tensor dv dt '[b, 4]) -> Tgt (T.Tensor dv dt '[b, 3])
irisPredict mmp = runFullModel irisModel . (,mmp)

labelToIrisClass :: (T.StandardDTypeValidation dv dt) => T.Tensor dv dt '[b, 3] -> [IrisClass]
labelToIrisClass = map toEnum . asValue . T.toDynamic . T.argmax @1 @T.DropDim

irisPredict' :: (KnownNat b, SaneDT dv dt, T.StandardDTypeValidation dv dt, T.KnownDevice dv) => IParams dv dt -> Inp (T.Tensor dv dt '[b, 4]) -> [IrisClass]
irisPredict' mmp = labelToIrisClass . irisPredict mmp

irisAccuracy ::
  ( SaneDT dv dt,
    T.StandardDTypeValidation dv dt,
    T.KnownDevice dv
  ) =>
  IParams dv dt ->
  Double
irisAccuracy mmp = fromIntegral correct / fromIntegral total
  where
    -- run predictions and get ground truth labels for each batch
    batched = flip map irisTargets $ \(inp, tgt) ->
      let predicted = labelToIrisClass (irisPredict mmp inp)
          actual = labelToIrisClass tgt -- argmax of one-hot targets
       in zipWith (==) predicted actual

    results = concat batched
    correct = length (filter id results)
    total = length results

type Device = '(T.CPU, 0)

irisRes :: IO [(Double, Double)]
irisRes = map ((asValue . T.toDynamic . irisError) &&& irisAccuracy) <$> irisBestParams @T.Double @Device

epochsToAcc :: Double -> [Double] -> Int
epochsToAcc r = fst . fromJust . L.find ((>= r) . snd) . zip [0 ..]