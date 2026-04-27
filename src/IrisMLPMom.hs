{-# LANGUAGE TupleSections #-}

module IrisMLPMom where

import IrisData (IrisClass, irisClass, iris)
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
import Control.Monad

--

type MMPMom = (MMP, MMP)
type IParams = (MMPMom, MMPMom)

irisModel :: ParaLens' (Inp RV, IParams) () (Out RV)
irisModel = argToPara .#. m .#. m
  where
    m :: ParaLens' (MMP, MMP) RV RV
    m = matMulLensMom 0.7 . sigmoid

irisModelLoss :: ParaLens' ((Inp RV, IParams), Tgt RV) () (Out R)
irisModelLoss = irisModel .#. lossSmooth

irisModel' :: ParaLRLens' ((Inp RV, IParams), Tgt RV) ()
irisModel' = irisModel .#. lossSmooth . lrSmooth 0.01

irisEpoch :: IParams -> IParams
irisEpoch mmp = trainMany irisModel' mmp irisTargets

blankMMP :: Int -> Int -> MMP
blankMMP i o = (konst 0 (o,i), konst 0 o)

initParamsMom :: Int -> Int -> IO MMPMom
initParamsMom i o = (blankMMP i o,) <$> initParams i o

irisInitParams :: IO IParams
irisInitParams = liftA2 (,) (initParamsMom 4 3) (initParamsMom 3 3)

irisBestParams :: IO [IParams]
irisBestParams = iterate irisEpoch <$> irisInitParams

-- irisBestParams :: IO [IParams]
-- -- start with the same params twice, ideally could maybe do
-- -- Maybe on the prev but shouldn't be meaningfully different
-- irisBestParams = iterate irisEpoch . join (,) <$> irisInitParamsMLP

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