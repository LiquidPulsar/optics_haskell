{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE TypeSynonymInstances #-}

module IrisNoLens where

import GHC.Generics
import Control.Monad (foldM)

import Torch
import qualified Torch.Functional as F
import Control.DeepSeq

--------------------------------------------------------------------------------
-- PARAMETERS
--------------------------------------------------------------------------------

data IrisModelSpec = IrisModelSpec
  { inputFeatures  :: Int
  , outputFeatures :: Int
  }

data IrisModel = IrisModel
  { linearLayer :: Linear
  }
  deriving (Generic, Show, NFData)

instance NFData Parameter where

instance NFData Linear where
  rnf (Linear weight bias) =
    rnf weight `seq`
    rnf bias

instance Parameterized IrisModel

instance Randomizable IrisModelSpec IrisModel where
  sample IrisModelSpec{..} =
    IrisModel
      <$> sample
            (LinearSpec inputFeatures outputFeatures)

--------------------------------------------------------------------------------
-- MODEL
--------------------------------------------------------------------------------

-- equivalent to:
--
-- matMulLens . sigmoid
--
irisModel
  :: IrisModel
  -> Tensor
  -> Tensor
irisModel IrisModel{..} input =
  sigmoid $
    linear linearLayer input

--------------------------------------------------------------------------------
-- LOSS
--------------------------------------------------------------------------------

-- equivalent to:
--
-- irisModel .#. lossSmooth
--
irisModelLoss
  :: IrisModel
  -> Tensor
  -> Tensor
  -> Tensor
irisModelLoss model input target =
  binaryCrossEntropyLoss'
    target
    prediction
  where
    prediction =
      irisModel model input

--------------------------------------------------------------------------------
-- TRAIN STEP
--------------------------------------------------------------------------------

-- equivalent to:
--
-- irisModel .#. lossSmooth . lrSmooth 0.01
--
irisTrainStep
  :: Optimizer o
  => IrisModel
  -> o
  -> Tensor
  -> Tensor
  -> IO IrisModel
irisTrainStep model optimizer input target = do

  let loss =
        irisModelLoss
          model
          input
          target

  (newModel, _) <-
    runStep
      model
      optimizer
      loss
      1e-2

  pure newModel

--------------------------------------------------------------------------------
-- TRAIN MANY
--------------------------------------------------------------------------------

-- equivalent to:
--
-- trainMany irisModel' params irisTargets
--
irisEpoch
  :: Optimizer o
  => IrisModel
  -> o
  -> [(Tensor, Tensor)]
  -> IO IrisModel
irisEpoch initModel optimizer dataset =
  foldM step initModel dataset
  where
    step model (input, target) =
      irisTrainStep
        model
        optimizer
        input
        target

--------------------------------------------------------------------------------
-- FULL TRAINING
--------------------------------------------------------------------------------

-- equivalent to:
--
-- iterate irisEpoch <$> irisInitParams
--
irisTrain
  :: Optimizer o
  => Int
  -> IrisModel
  -> o
  -> [(Tensor, Tensor)]
  -> IO IrisModel
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
irisPredict
  :: IrisModel
  -> Tensor
  -> Tensor
irisPredict =
  irisModel

--------------------------------------------------------------------------------
-- CLASS PREDICTION
--------------------------------------------------------------------------------

-- equivalent to:
--
-- labelToIrisClass
--
irisPredictClass
  :: IrisModel
  -> Tensor
  -> [Int]
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

irisAccuracy
  :: IrisModel
  -> [(Tensor, Tensor)]
  -> Double
irisAccuracy model dataset =
  fromIntegral correct
    / fromIntegral total
  where

    batchResults =
      flip map dataset $
        \(input, target) ->

          let predicted =
                asValue @[[Int]] $
                  argmax
                    (Dim 1)
                    RemoveDim
                    (irisPredict model input)

              actual =
                asValue @[[Int]] $
                  argmax
                    (Dim 1)
                    RemoveDim
                    target

           in zipWith
                (==)
                (concat predicted)
                (concat actual)

    results =
      concat batchResults

    correct =
      length $
        filter id results

    total =
      length results

--------------------------------------------------------------------------------
-- INITIALIZATION
--------------------------------------------------------------------------------

irisInitModel
  :: IO IrisModel
irisInitModel =
  sample $
    IrisModelSpec
      4
      3