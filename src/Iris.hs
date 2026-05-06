{-# LANGUAGE TupleSections #-}

module Iris where

import IrisData (Iris, IrisClass (..), iris, irisClass, petalLength, petalWidth, sepalLength, sepalWidth)
import Loss
import Types
import Core
import Optim
import Numeric.LinearAlgebra
import qualified Data.Vector.Storable as VS
import Control.Arrow
import Models
import Data.Functor
import Data.Function
import Layers
import qualified Data.List as L
import Data.Maybe


irisToVec :: Iris -> RV
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

oneHot :: Int -> Int -> RV
oneHot s i = konst 0 s VS.// [(i, 1)]

irisClassToLabel :: IrisClass -> RV
irisClassToLabel = oneHot 3 . fromEnum

labelToIrisClass :: RV -> IrisClass
labelToIrisClass = toEnum . maxIndex

irisToLabel :: Iris -> RV
irisToLabel = irisClassToLabel . irisClass

irisTargets :: [(RV, RV)]
irisTargets = map (irisToVec &&& irisToLabel) iris

--

type IParams = MMP

irisModel :: ParaLens' (Inp RV, IParams) () (Out RV)
irisModel = argToPara .#. matMulLens . sigmoid

irisModelLoss :: ParaLens' ((Inp RV, IParams), Tgt RV) () (Out R)
irisModelLoss = irisModel .#. lossSmooth

irisModel' :: ParaLRLens' ((Inp RV, IParams), Tgt RV) ()
irisModel' = irisModel .#. lossSmooth . lrSmooth 0.01

irisEpoch :: IParams -> IParams
irisEpoch mmp = trainMany irisModel' mmp irisTargets

initParams :: Int -> Int -> IO (Matrix Double, Vector Double)
initParams inputDim outputDim = do
  w <- rand outputDim inputDim
  b <- flatten <$> rand outputDim 1
  let f x = scale 0.01 (x - 0.5)
  return (f w, f b)

irisInitParams :: IO IParams
irisInitParams = initParams 4 3

irisBestParams :: IO [IParams]
irisBestParams = iterate irisEpoch <$> irisInitParams

irisGetEpoch :: Int -> IO IParams
irisGetEpoch i = irisBestParams <&> (!! i)

irisError :: IParams -> R
irisError = sum . flip map irisTargets . runOne irisModelLoss

irisPredict :: IParams -> Inp RV -> Tgt RV
irisPredict mmp = runFullModel irisModel . (,mmp)

irisPredict' :: IParams -> Inp RV -> IrisClass
irisPredict' mmp = labelToIrisClass . irisPredict mmp

irisAccuracy :: IParams -> R
irisAccuracy mmp = sum tgts / fromIntegral (length tgts)
  where
    predict = irisPredict' mmp . irisToVec
    acc x y = fromIntegral . fromEnum $ x == y
    tgts = map (liftA2 acc predict irisClass) iris

irisRes :: IO [(R, R)]
irisRes = map (irisError &&& irisAccuracy) <$> irisBestParams

epochsToAcc :: R -> [R] -> Int
epochsToAcc r = fst . fromJust . L.find ((>=r) . snd) . zip [0..]