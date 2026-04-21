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

type IParams = (MMP, MMP)

irisModel :: ParaLens' (Inp RV, IParams) () (Out RV)
irisModel = argToPara .#. matMulLens . sigmoid .#. matMulLens . sigmoid

irisModelLoss :: ParaLens' ((Inp RV, IParams), Tgt RV) () (Out R)
irisModelLoss = irisModel .#. lossSmooth

irisModel' :: ParaLRLens' ((Inp RV, IParams), Tgt RV) ()
irisModel' = irisModel .#. lossSmooth . lrSmooth 0.01

irisEpoch :: IParams -> IParams
irisEpoch mmp = trainMany irisModel' mmp irisTargets

irisInitParams :: IO IParams
irisInitParams = liftA2 (,) (initParams 4 3) (initParams 3 3)


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