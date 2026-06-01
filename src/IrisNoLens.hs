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
import Control.Exception (evaluate)
import Control.Monad (foldM)
import Data.List (foldl')
import GHC.Generics
import Torch hiding (step)

--------------------------------------------------------------------------------
-- PARAMETERS
--------------------------------------------------------------------------------

data IrisModelSpec = IrisModelSpec
  { inputFeatures :: Int,
    hiddenFeatures :: Int,
    outputFeatures :: Int
  }

data IrisModel = IrisModel { linearLayer1 :: Linear, linearLayer2 :: Linear }
  deriving (Generic, Show, NFData)

instance NFData Parameter

deriving instance NFData Linear -- where
  -- rnf (Linear weight bias) =
  --   rnf weight `seq` rnf bias

instance Parameterized IrisModel

instance Randomizable IrisModelSpec IrisModel where
  sample IrisModelSpec {..} =
    IrisModel <$> sample (LinearSpec inputFeatures hiddenFeatures) <*> sample (LinearSpec hiddenFeatures outputFeatures)

--------------------------------------------------------------------------------
-- MODEL
--------------------------------------------------------------------------------

-- equivalent to:
--
-- matMulLens . sigmoid
--
irisModel :: IrisModel -> Tensor -> Tensor
irisModel IrisModel {..} = sigmoid . linear linearLayer2 . sigmoid . linear linearLayer1

--------------------------------------------------------------------------------
-- LOSS
--------------------------------------------------------------------------------

-- equivalent to:
--
-- irisModel .#. lossSmooth
--
irisModelLoss :: IrisModel -> Tensor -> Tensor -> Tensor
-- irisModelLoss model input target = binaryCrossEntropyLoss' target $ irisModel model input
irisModelLoss model input target = mseLoss target $ irisModel model input

--------------------------------------------------------------------------------
-- TRAIN STEP
--------------------------------------------------------------------------------

-- equivalent to:
--
-- irisModel .#. lossSmooth . lrSmooth 0.01
--
irisTrainStep :: Optimizer o => IrisModel -> o -> Tensor -> Tensor -> IO IrisModel
irisTrainStep model optimizer input target = fst <$> runStep model optimizer loss 1e-2
  where loss = irisModelLoss model input target

--------------------------------------------------------------------------------
-- TRAIN MANY
--------------------------------------------------------------------------------

-- equivalent to:
--
-- trainMany irisModel' params irisTargets
--
irisEpoch :: Optimizer o => IrisModel -> o -> [(Tensor, Tensor)] -> IO IrisModel
irisEpoch initModel optimizer = foldM step initModel
  where step model = uncurry $ irisTrainStep model optimizer

-- Strict left-fold variant: same per-batch work, different fold structure.
-- Isolates foldM overhead by removing the right-recursive >>=  chain.
irisEpochFoldl :: Optimizer o => IrisModel -> o -> [(Tensor, Tensor)] -> IO IrisModel
irisEpochFoldl initModel optimizer =
  foldl' (\mio b -> mio >>= \m -> uncurry (irisTrainStep m optimizer) b)
         (return initModel)

-- Forward pass only: compute the loss but do not call runStep.
-- No .backward(), no flattenParameters, no parameter write-back.
-- Used to isolate the combined cost of autograd backward + Generic traversal + update.
irisEpochForwardOnly :: IrisModel -> [(Tensor, Tensor)] -> IO ()
irisEpochForwardOnly model =
  mapM_ (\(inp, tgt) -> evaluate $ irisModelLoss model inp tgt)

--------------------------------------------------------------------------------
-- FULL TRAINING
--------------------------------------------------------------------------------

-- equivalent to:
--
-- iterate irisEpoch <$> irisInitParams
--
irisTrain ::
  Optimizer o =>
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
irisInitModel = sample $ IrisModelSpec 4 4 3