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

convert :: forall device dtype shape. (T.KnownDevice device, T.KnownDType dtype) => Tensor -> T.Tensor device dtype shape
convert =
  T.UnsafeMkTensor
    . toDevice (T.deviceVal @device)
    . toType (T.dtypeVal @dtype)
{-# INLINE convert #-} -- already inlines but let's be safe

irisToTensor ::
  ( T.KnownDType dtype,
    T.KnownDevice device
  ) =>
  [Iris] ->
  ( T.Tensor device dtype '[NumIris, 4],
    T.Tensor device dtype '[NumIris, 3]
  )
irisToTensor = feats &&& classes
  where
    feats = convert . asTensor . map irisToVec
    classes = convert . oneHot 3 . asTensor . map (fromEnum . irisClass)

sliceBatch ::
  forall batchSize features n device dtype.
  ( KnownNat batchSize,
    KnownNat features
  ) =>
  Int -> -- runtime offset
  T.Tensor device dtype '[n, features] ->
  T.Tensor device dtype '[batchSize, features]
sliceBatch offset t = T.UnsafeMkTensor $ narrow_tlll (T.toDynamic t) 0 offset b
  where
    b = T.natValI @batchSize

batches :: forall n batch device dtype features. (T.All KnownNat [batch, features, n]) => T.Tensor device dtype '[n, features] -> [T.Tensor device dtype '[batch, features]]
batches dataset = [sliceBatch @batch (i * b) dataset | i <- [0 .. (n `div` b) - 1]]
  where
    b = T.natValI @batch
    n = T.natValI @n

irisTargets :: (T.KnownDType dtype, T.KnownDevice device) => [(T.Tensor device dtype '[BatchSize, 4], T.Tensor device dtype '[BatchSize, 3])]
irisTargets = zip (batches inps) (batches tgts)
  where
    (inps, tgts) = irisToTensor iris

type IParams device dtype = MMP device dtype 3 4

type SaneDT device dtype =
  ( T.KnownDType dtype,
    T.StandardFloatingPointDTypeValidation device dtype,
    CanMMLens device dtype
  )

irisModel ::
  ( SaneDT device dtype,
    T.KnownDevice device
  ) =>
  ParaLens' (Inp (T.Tensor device dtype '[b, 4]), IParams device dtype) () (Out (T.Tensor device dtype '[b, 3]))
irisModel = argToPara .#. matMulLens . sigmoid
{-# INLINE irisModel #-}

test :: (SaneDT device dtype, T.KnownDevice device) => (Inp (T.Tensor device dtype '[b, 4]), IParams device dtype) -> Out (T.Tensor device dtype '[b, 3])
test = runFullModel irisModel

handRolledTest :: (SaneDT device dtype, T.KnownDevice device) => (Inp (T.Tensor device dtype '[b, 4]), IParams device dtype) -> Out (T.Tensor device dtype '[b, 3])
handRolledTest (i, (m, b)) =
  let res = T.matmul i (transp m)
      res' = T.add res b
      res'' = T.sigmoid res'
   in res''

irisModelLoss ::
  ( SaneDT device dtype,
    T.KnownDevice device
  ) =>
  ParaLens' ((Inp (T.Tensor device dtype '[b, 4]), IParams device dtype), Tgt (T.Tensor device dtype '[b, 3])) () (Out (T.Tensor device dtype '[]))
irisModelLoss = irisModel .#. lossSmooth
{-# INLINE irisModelLoss #-}

irisModel' ::
  ( SaneDT device dtype,
    T.KnownDevice device
  ) =>
  ParaLens' ((Inp (T.Tensor device dtype '[b, 4]), IParams device dtype), Tgt (T.Tensor device dtype '[b, 3])) () ()
irisModel' = irisModel .#. lossSmooth . lrSmooth 0.01
{-# INLINE irisModel' #-}

irisEpoch :: (SaneDT device dtype, T.KnownDevice device) => IParams device dtype -> IParams device dtype
irisEpoch mmp = trainMany irisModel' mmp irisTargets

-- initParams :: Int -> Int -> IO (Matrix Double, Vector Double)
-- initParams inputDim outputDim = do
--   w <- rand outputDim inputDim
--   b <- flatten <$> rand outputDim 1
--   let f x = scale 0.01 (x - 0.5)
--   return (f w, f b)

irisInitParams :: (T.KnownDType dtype, T.RandDTypeIsValid device dtype, T.KnownDevice device) => IO (IParams device dtype)
irisInitParams = liftA2 (,) T.randn T.randn

irisBestParams :: (T.KnownDType dtype, T.RandDTypeIsValid device dtype, SaneDT device dtype, T.KnownDevice device) => IO [IParams device dtype]
irisBestParams = iterate irisEpoch <$> irisInitParams

irisGetEpoch :: (T.KnownDType dtype, T.RandDTypeIsValid device dtype, SaneDT device dtype, T.KnownDevice device) => Int -> IO (IParams device dtype)
irisGetEpoch i = irisBestParams <&> (!! i)

irisError :: (SaneDT device dtype, T.KnownDevice device) => IParams device dtype -> T.Tensor device dtype '[]
irisError = sum . flip map irisTargets . runOne irisModelLoss

irisPredict :: (KnownNat b, SaneDT device dtype, T.KnownDevice device) => IParams device dtype -> Inp (T.Tensor device dtype '[b, 4]) -> Tgt (T.Tensor device dtype '[b, 3])
irisPredict mmp = runFullModel irisModel . (,mmp)

labelToIrisClass :: (T.StandardDTypeValidation device dtype) => T.Tensor device dtype '[b, 3] -> [IrisClass]
labelToIrisClass = map toEnum . asValue . T.toDynamic . T.argmax @1 @T.DropDim

irisPredict' :: (KnownNat b, SaneDT device dtype, T.StandardDTypeValidation device dtype, T.KnownDevice device) => IParams device dtype -> Inp (T.Tensor device dtype '[b, 4]) -> [IrisClass]
irisPredict' mmp = labelToIrisClass . irisPredict mmp

irisAccuracy ::
  ( SaneDT device dtype,
    T.StandardDTypeValidation device dtype,
    T.KnownDevice device
  ) =>
  IParams device dtype ->
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