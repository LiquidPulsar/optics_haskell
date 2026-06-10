{-# LANGUAGE DataKinds #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# OPTIONS_GHC -Wno-orphans #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
-- {-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE BangPatterns #-}
-- {-# LANGUAGE MultiParamTypeClasses #-}


import Control.DeepSeq
import Control.Exception (evaluate)
import Criterion.Main
import qualified Iris as I
import qualified IrisNoLens as HT
import Static.Iris
import Static.Mnist
import Torch
import qualified Torch.Typed as T

type TestDevice = '(T.CPU, 0)
type IrisDType = T.Float

htIrisTargets :: [(Tensor, Tensor)]
htIrisTargets = [(T.toDynamic l, T.toDynamic r) | (l, r) <- irisTargets @TestDevice @IrisDType ]

main :: IO ()
main = do
  params <- irisGetEpoch @1 @TestDevice @IrisDType 100
  paramsOld <- I.irisGetEpoch 100
  initTorch <- HT.irisInitModel
  paramsTorch <- HT.irisTrain 100 initTorch GD htIrisTargets

  (someInput, _) : _ <- pure $ irisTargets @TestDevice @IrisDType 
  (someInputOld, _) : _ <- pure I.irisTargets
  (someInputTorch, _) : _ <- pure htIrisTargets

  mnistPF <- mnistInitParams @TestDevice @T.Float
  trainTF <- uncurry mnistTargets <$>
    loadMnist @NumTrain @TestDevice @T.Float
      "data/train-images-idx3-ubyte"
      "data/train-labels-idx1-ubyte"

  mnistPD <- mnistInitParams @TestDevice @T.Double
  trainTD <- uncurry mnistTargets <$>
    loadMnist @NumTrain @TestDevice @T.Double
      "data/train-images-idx3-ubyte"
      "data/train-labels-idx1-ubyte"

  -- Warm up LibTorch kernel JIT and MKL auto-tuning for both dtypes
  -- so neither benchmark pays the cold-start tax.
  !_ <- evaluate . force $ mnistEpoch trainTF mnistPF
  !_ <- evaluate . force $ mnistEpoch trainTD mnistPD

  defaultMain
    [ -- bgroup
      --   "accuracy"
      --   [ bench "typed-hasktorch" $
      --       nf irisAccuracy params,
      --     bench "hmatrix" $
      --       nf I.irisAccuracy paramsOld,
      --     bench "dynamic-hasktorch" $
      --       nf (HT.irisAccuracy paramsTorch) htIrisTargets
      --   ],
      -- bgroup
      --   "epoch"
      --   [ bench "typed-hasktorch" $
      --       nf (irisEpoch @1) params,
      --     bench "hmatrix" $
      --       nf I.irisEpoch paramsOld,
      --     bench "dynamic-hasktorch" $
      --       nfIO $
      --         HT.irisEpoch paramsTorch GD htIrisTargets
      --   ],
      -- bgroup
      --   "raw_fwd"
      --   [ bench "typed-hasktorch" $
      --       nf test (someInput, params),
      --     bench "typed-hasktorch-handroll" $
      --       nf handRolledTest (someInput, params)
      --   ],
      -- Overhead isolation: measures individual contributors to the dynamic epoch cost.
      --   fold-foldM      baseline (same as epoch/dynamic-hasktorch)
      --   fold-foldl'     replaces foldM with explicit foldl' chain; diff = fold-structure overhead
      --   forward-only    no runStep at all; diff from baseline = backward + flattenParams + update
      --   flattenParams   Generic traversal alone x18; diff from forward-only ~ traversal share
      -- bgroup
      --   "overhead-breakdown"
      --   [ bench "fold-foldM" $
      --       nfIO $
      --         HT.irisEpoch initTorch GD htIrisTargets,
      --     bench "fold-foldl'" $
      --       nfIO $
      --         HT.irisEpochFoldl initTorch GD htIrisTargets,
      --     bench "forward-only" $
      --       nfIO $
      --         HT.irisEpochForwardOnly initTorch htIrisTargets,
      --     bench "flattenParams-x18" $
      --       nf (\() -> map (const (flattenParameters initTorch)) htIrisTargets) ()
      --   ],
      bgroup
        "mnist-epoch"
        [ bench "float" $ nf (mnistEpoch trainTF) mnistPF,
          bench "double" $ nf (mnistEpoch trainTD) mnistPD
        ]
      -- bgroup
      --   "predict"
      --   [ bench "typed-hasktorch" $
      --       nf (irisPredict' params) someInput,
      --     bench "typed-hasktorch-handroll" $
      --       nf (labelToIrisClass . test @IrisDevice @IrisDType) (someInput, params),
      --     bench "hmatrix" $
      --       nf (I.irisPredict' paramsOld) someInputOld,
      --     bench "dynamic-hasktorch" $
      --       nf (HT.irisPredictClass paramsTorch) someInputTorch
      --   ]
    ]

--------------------------------------------------------------------------------
-- NFData instances
--------------------------------------------------------------------------------

instance NFData (T.Tensor dv dt shape) where
  rnf :: T.Tensor dv dt shape -> ()
  rnf = rnf . T.toDynamic

--------------------------------------------------------------------------------
-- Time-to-accuracy helpers
--------------------------------------------------------------------------------

-- Train from a fresh random init until accuracy >= threshold; return epoch count.
-- Separate functions per dtype because RandStack lives in an unexposed module.
trainToAcc :: forall dt . (SaneDT TestDevice dt, T.KnownDType dt, T.RandDTypeIsValid TestDevice dt) => Double -> IO Int
trainToAcc thres = do
  epochs <- irisBestParams @1 @TestDevice @dt
  return $! epochsToAcc thres $ map (irisAccuracy @1) epochs