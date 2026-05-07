{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# OPTIONS_GHC -Wno-orphans #-}
{-# LANGUAGE StandaloneDeriving #-}

module IrisNoLens where

import Control.DeepSeq
import Control.Monad (foldM)
import GHC.Generics
import Torch hiding (step)

--------------------------------------------------------------------------------
-- PARAMETERS
--------------------------------------------------------------------------------

data IrisModelSpec = IrisModelSpec
  { inputFeatures :: Int,
    outputFeatures :: Int
  }

newtype IrisModel = IrisModel { linearLayer :: Linear }
  deriving (Generic, Show, NFData)

instance NFData Parameter

deriving instance NFData Linear -- where
  -- rnf (Linear weight bias) =
  --   rnf weight `seq` rnf bias

instance Parameterized IrisModel

instance Randomizable IrisModelSpec IrisModel where
  sample IrisModelSpec {..} =
    IrisModel <$> sample (LinearSpec inputFeatures outputFeatures)

--------------------------------------------------------------------------------
-- MODEL
--------------------------------------------------------------------------------

-- equivalent to:
--
-- matMulLens . sigmoid
--
irisModel :: IrisModel -> Tensor -> Tensor
irisModel IrisModel {..} = sigmoid . linear linearLayer

--------------------------------------------------------------------------------
-- LOSS
--------------------------------------------------------------------------------

-- equivalent to:
--
-- irisModel .#. lossSmooth
--
irisModelLoss :: IrisModel -> Tensor -> Tensor -> Tensor
irisModelLoss model input target = binaryCrossEntropyLoss' target $ irisModel model input

--------------------------------------------------------------------------------
-- TRAIN STEP
--------------------------------------------------------------------------------

-- equivalent to:
--
-- irisModel .#. lossSmooth . lrSmooth 0.01
--
irisTrainStep :: (Optimizer o) => IrisModel -> o -> Tensor -> Tensor -> IO IrisModel
irisTrainStep model optimizer input target = fst <$> runStep model optimizer loss 1e-2
  where loss = irisModelLoss model input target

--------------------------------------------------------------------------------
-- TRAIN MANY
--------------------------------------------------------------------------------

-- equivalent to:
--
-- trainMany irisModel' params irisTargets
--
irisEpoch :: (Optimizer o) => IrisModel -> o -> [(Tensor, Tensor)] -> IO IrisModel
irisEpoch initModel optimizer = foldM step initModel
  where step model = uncurry $ irisTrainStep model optimizer

--------------------------------------------------------------------------------
-- FULL TRAINING
--------------------------------------------------------------------------------

-- equivalent to:
--
-- iterate irisEpoch <$> irisInitParams
--
irisTrain ::
  (Optimizer o) =>
  Int ->
  IrisModel ->
  o ->
  [(Tensor, Tensor)] ->
  IO IrisModel
irisTrain epochs initModel optimizer dataset =
  foldM step initModel [1 .. epochs]
  where
    step model _ =
      irisEpoch
        model
        optimizer
        dataset

--------------------------------------------------------------------------------
-- PREDICTION
--------------------------------------------------------------------------------

-- equivalent to:
--
-- runFullModel irisModel . (, params)
--
irisPredict ::
  IrisModel ->
  Tensor ->
  Tensor
irisPredict =
  irisModel

--------------------------------------------------------------------------------
-- CLASS PREDICTION
--------------------------------------------------------------------------------

-- equivalent to:
--
-- labelToIrisClass
--
irisPredictClass ::
  IrisModel ->
  Tensor ->
  [Int]
irisPredictClass model input =
  asValue $
    argmax
      (Dim 1)
      RemoveDim
      prediction
  where
    prediction =
      irisPredict
        model
        input

--------------------------------------------------------------------------------
-- ACCURACY
--------------------------------------------------------------------------------

irisAccuracy :: IrisModel -> [(Tensor, Tensor)] -> Double
irisAccuracy model dataset = fromIntegral correct / fromIntegral total
  where
    batchResults =
      flip map dataset $
        \(input, target) ->
          let parse = concat . asValue @[[Int]] . argmax (Dim 1) RemoveDim
           in zipWith (==) (parse (irisPredict model input)) (parse target)

    results = concat batchResults
    correct = length $ filter id results
    total = length results

--------------------------------------------------------------------------------
-- INITIALIZATION
--------------------------------------------------------------------------------

irisInitModel :: IO IrisModel
irisInitModel = sample $ IrisModelSpec 4 3