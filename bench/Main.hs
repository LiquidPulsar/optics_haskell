{-# LANGUAGE DataKinds #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -Wno-orphans #-}

import Control.DeepSeq
import Criterion.Main
import qualified Iris as I
import qualified IrisNoLens as HT
import Static.Iris
import Torch
import qualified Torch.Typed as T

type IrisDevice = '(T.CPU, 0)

type IrisDType = T.Float

htIrisTargets :: [(Tensor, Tensor)]
htIrisTargets = [(T.toDynamic l, T.toDynamic r) | (l, r) <- irisTargets @IrisDType @IrisDevice]

main :: IO ()
main = do
  params <- irisGetEpoch @IrisDType @IrisDevice 100
  paramsOld <- I.irisGetEpoch 100
  initTorch <- HT.irisInitModel
  paramsTorch <- HT.irisTrain 100 initTorch GD htIrisTargets

  (someInput, _) : _ <- pure irisTargets
  (someInputOld, _) : _ <- pure I.irisTargets
  (someInputTorch, _) : _ <- pure htIrisTargets

  defaultMain
    [ bgroup
        "accuracy"
        [ bench "typed-hasktorch" $
            nf irisAccuracy params,
          bench "hmatrix" $
            nf I.irisAccuracy paramsOld,
          bench "dynamic-hasktorch" $
            nf (HT.irisAccuracy paramsTorch) htIrisTargets
        ],
      bgroup
        "epoch"
        [ bench "typed-hasktorch" $
            nf irisEpoch params,
          bench "hmatrix" $
            nf I.irisEpoch paramsOld,
          bench "dynamic-hasktorch" $
            nfIO $
              HT.irisEpoch paramsTorch GD htIrisTargets
        ],
      bgroup
        "raw_fwd"
        [ bench "typed-hasktorch" $
            nf (irisPredict' params) someInput,
          bench "typed-hasktorch-handroll" $
            nf (test @IrisDevice @IrisDType) (someInput, params)
        ],
      bgroup
        "predict"
        [ bench "typed-hasktorch" $
            nf (irisPredict' params) someInput,
          bench "typed-hasktorch-handroll" $
            nf (labelToIrisClass . test @IrisDevice @IrisDType) (someInput, params),
          bench "hmatrix" $
            nf (I.irisPredict' paramsOld) someInputOld,
          bench "dynamic-hasktorch" $
            nf (HT.irisPredictClass paramsTorch) someInputTorch
        ]
    ]

--------------------------------------------------------------------------------
-- NFData instances
--------------------------------------------------------------------------------

instance NFData (T.Tensor device dtype shape) where
  rnf :: T.Tensor device dtype shape -> ()
  rnf = rnf . T.toDynamic