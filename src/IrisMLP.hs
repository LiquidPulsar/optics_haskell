{-# LANGUAGE TupleSections #-}

module IrisMLP where

import Numeric.Datasets.Iris (IrisClass, irisClass, iris)
import Loss
import Types
import Core
import Optim
import Numeric.LinearAlgebra
import Control.Arrow
import Models
import Control.Applicative
import Data.Functor
import Layers
import Iris (irisTargets, labelToIrisClass, irisToVec, initParams)

--

type IrisParams = (MMP, MMP)

irisModel :: ParaLens' (Inp RVector, IrisParams) () (Out RVector)
irisModel = argToPara .#. matMulLens . sigmoid .#. matMulLens . sigmoid

irisModelLoss :: ParaLens' ((Inp RVector, IrisParams), Tgt RVector) () (Out R)
irisModelLoss = irisModel .#. lossSmooth

irisModel' :: ParaLRLens' ((Inp RVector, IrisParams), Tgt RVector) ()
irisModel' = irisModel .#. lossSmooth . lrSmooth 0.01

irisEpoch :: IrisParams -> IrisParams
irisEpoch mmp = trainMany irisModel' mmp irisTargets

irisInitParamsMLP :: IO IrisParams
irisInitParamsMLP = liftA2 (,) (initParams 4 3) (initParams 3 3)


irisBestParams :: IO [IrisParams]
irisBestParams = iterate irisEpoch <$> irisInitParamsMLP

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