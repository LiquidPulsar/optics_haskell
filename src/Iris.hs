{-# LANGUAGE TupleSections #-}

module Iris where

import Numeric.Datasets.Iris (Iris, IrisClass (..), iris, irisClass, petalLength, petalWidth, sepalLength, sepalWidth)
import Loss
import Types
import Core
import Optim
import Numeric.LinearAlgebra
import qualified Data.Vector.Storable as VS
import Control.Arrow
import Models
import Control.Applicative
import Data.Functor
import Data.Function
import Layers


irisToVec :: Iris -> RVector
irisToVec =
  fromList
    . flip
      map
      [ sepalLength,
        sepalWidth,
        petalLength,
        petalWidth
      ]
    . (&)

oneHot :: Int -> Int -> RVector
oneHot s i = konst 0 s VS.// [(i, 1)]

irisClassToLabel :: IrisClass -> RVector
irisClassToLabel = oneHot 3 . fromEnum

labelToIrisClass :: RVector -> IrisClass
labelToIrisClass = toEnum . maxIndex

irisToLabel :: Iris -> RVector
irisToLabel = irisClassToLabel . irisClass

irisTargets :: [(RVector, RVector)]
irisTargets = map (irisToVec &&& irisToLabel) iris

--

type IrisParams = MMP

irisModel :: ParaLens' (Inp RVector, IrisParams) () (Out RVector)
irisModel = argToPara .#. matMulLens . sigmoid

irisModelLoss :: ParaLens' ((Inp RVector, IrisParams), Tgt RVector) () (Out R)
irisModelLoss = irisModel .#. lossSmooth

irisModel' :: ParaLRLens' ((Inp RVector, IrisParams), Tgt RVector) ()
irisModel' = irisModel .#. lossSmooth . lrSmooth 0.01

irisEpoch :: IrisParams -> IrisParams
irisEpoch mmp = trainMany irisModel' mmp irisTargets

irisInitParams :: IO IrisParams
irisInitParams = do
  w <- rand 3 4
  b <- flatten <$> rand 3 1
  let f x = scale 0.01 (x - 0.5)
  return (f w, f b)

irisBestParams :: IO [IrisParams]
irisBestParams = iterate irisEpoch <$> irisInitParams

irisGetEpoch :: Int -> IO IrisParams
irisGetEpoch i = irisBestParams <&> (!! i)

irisError :: IrisParams -> R
irisError = sum . flip map irisTargets . runOne irisModelLoss

irisPredict :: IrisParams -> Inp RVector -> Tgt RVector
irisPredict mmp = runFullModel irisModel . (,mmp)

irisPredict' :: IrisParams -> Inp RVector -> IrisClass
irisPredict' mmp = labelToIrisClass . irisPredict mmp

irisAccuracy :: IrisParams -> R
irisAccuracy mmp = sum tgts / fromIntegral (length tgts)
  where
    predict = irisPredict' mmp . irisToVec
    acc x y = fromIntegral . fromEnum $ x == y
    tgts = map (liftA2 acc predict irisClass) iris

irisRes :: IO [(R, R)]
irisRes = map (irisError &&& irisAccuracy) <$> irisBestParams