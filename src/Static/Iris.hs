{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}

module Static.Iris where

import Control.Arrow
-- import Types
import Core
import Data.Function
import IrisData
import Static.Layers ( sigmoid, matMulLens, MMP, CanMMLens )
import Torch (TensorLike (asTensor), oneHot, asValue)
import qualified Torch.Typed as T
import Torch.Functional.Internal (narrow_tlll)
import Types (Inp, Out, Tgt)
import GHC.TypeLits
import Static.Loss
import Static.Optim
import Models (trainMany, runOne, runFullModel)
import Data.Data
import Data.Functor
import Data.Maybe
import qualified Data.List as L

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

irisToTensor :: [Iris] -> (T.Tensor device dtype '[NumIris, 4], T.Tensor device dtype '[NumIris, 3])
irisToTensor = feats &&& classes
  where
    -- TODO: match devices
    feats = T.UnsafeMkTensor . asTensor . map irisToVec
    classes = T.UnsafeMkTensor . oneHot 3 . asTensor . map (fromEnum . irisClass)

sliceBatch
  :: forall batchSize features n device dtype
   . ( KnownNat batchSize
     , KnownNat features
     )
  => Int                                          -- runtime offset
  -> T.Tensor device dtype '[n, features]
  -> T.Tensor device dtype '[batchSize, features]
sliceBatch offset t = T.UnsafeMkTensor $ narrow_tlll (T.toDynamic t) 0 offset b
  where b = fromIntegral . natVal $ Proxy @batchSize

batches :: forall n batch device dtype features . (T.All KnownNat [batch, features, n]) => T.Tensor device dtype '[n, features] -> [T.Tensor device dtype '[batch, features]]
batches dataset = [ sliceBatch @batch (i * b) dataset | i <- [0 .. (n `div` b) - 1] ]
  where
    b = fromIntegral . natVal $ Proxy @batch
    n = fromIntegral . natVal $ Proxy @n


irisTargets :: [(T.Tensor device dtype '[BatchSize, 4], T.Tensor device dtype '[BatchSize, 3])]
irisTargets = zip (batches inps) (batches tgts)
  where (inps, tgts) = irisToTensor iris

type IParams device dtype = MMP device dtype 3 4

type SaneDT device dtype = (
    T.StandardFloatingPointDTypeValidation device dtype
    , CanMMLens device dtype
  )

irisModel ::
  (t ~ T.Tensor device dtype
  , params ~ IParams device dtype
  , KnownNat b
  , SaneDT device dtype
  , T.KnownDevice device) =>
  ParaLens' (Inp (t '[b, 4]), params) () (Out (t '[b, 3]))
irisModel = argToPara .#. matMulLens . sigmoid

irisModelLoss ::
  (t ~ T.Tensor device dtype
  , mmp ~ MMP device dtype 3 4
  , KnownNat b
  , SaneDT device dtype
  , T.KnownDevice device) =>
  ParaLens' ((Inp (t '[b, 4]), mmp), Tgt (t '[b, 3])) () (Out (t '[]))
irisModelLoss = irisModel .#. lossSmooth

irisModel' ::
  forall t mmp device dtype b.
  (t ~ T.Tensor device dtype
  , mmp ~ MMP device dtype 3 4
  , KnownNat b
  , SaneDT device dtype
  , T.KnownDevice device) =>
  ParaLens' ((Inp (t '[b, 4]), mmp), Tgt (t '[b, 3])) () ()
irisModel' = irisModel .#. lossSmooth . lrSmooth 0.01

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

irisGetEpoch :: forall device dtype. (T.KnownDType dtype, T.RandDTypeIsValid device dtype, SaneDT device dtype, T.KnownDevice device) => Int -> IO (IParams device dtype)
irisGetEpoch i = irisBestParams @dtype <&> (!! i)

irisError :: (SaneDT device dtype, T.KnownDevice device) => IParams device dtype -> T.Tensor device dtype '[]
irisError = sum . flip map irisTargets . runOne irisModelLoss

irisPredict :: (t ~ T.Tensor device dtype, KnownNat b, SaneDT device dtype, T.KnownDevice device) => IParams device dtype -> Inp (t '[b, 4]) -> Tgt (t '[b, 3])
irisPredict mmp = runFullModel irisModel . (,mmp)

labelToIrisClass :: forall device dtype b . (T.StandardDTypeValidation device dtype) => T.Tensor device dtype '[b, 3] -> [IrisClass]
labelToIrisClass = map toEnum . asValue . T.toDynamic . T.argmax @1 @T.DropDim

irisPredict' :: (t ~ T.Tensor device dtype, KnownNat b, SaneDT device dtype, T.StandardDTypeValidation device dtype, T.KnownDevice device) => IParams device dtype -> Inp (t '[b, 4]) -> [IrisClass]
irisPredict' mmp = labelToIrisClass . irisPredict mmp

irisAccuracy
  :: ( SaneDT device dtype
     , T.StandardDTypeValidation device dtype
     , T.KnownDevice device
     )
  => IParams device dtype
  -> Double
irisAccuracy mmp = fromIntegral correct / fromIntegral total
  where
    -- run predictions and get ground truth labels for each batch
    batched = flip map irisTargets $ \(inp, tgt) ->
      let predicted = labelToIrisClass (irisPredict mmp inp)
          actual    = labelToIrisClass tgt   -- argmax of one-hot targets
      in  zipWith (==) predicted actual

    results  = concat batched
    correct  = length (filter id results)
    total    = length results

type Device = '(T.CPU, 0)

irisRes :: IO [(Double, Double)]
irisRes = map ((asValue . T.toDynamic . irisError) &&& irisAccuracy) <$> irisBestParams @T.Double @Device

epochsToAcc :: Double -> [Double] -> Int
epochsToAcc r = fst . fromJust . L.find ((>=r) . snd) . zip [0..]