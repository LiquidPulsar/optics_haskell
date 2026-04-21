{-# LANGUAGE TupleSections #-}
{-# LANGUAGE RankNTypes #-}

module Examples where

import Types
import Core
import Optim
import Numeric.LinearAlgebra
import Loss
import Control.Arrow
import Models
import Layers

-------------------------
-- MODELS --
-------------------------

example :: ParaLens' (Inp RV, (MMP, MMP)) () (Out RV)
example = argToPara .#. matMulLens . relu .#. matMulLens . relu

example' :: ParaLRLens' ((Inp RV, (MMP, MMP)), Tgt RV) ()
example' = example .#. lossSmooth . lrSmooth 1

targetV :: RV
targetV = fromList [1, 2, 3]

inputV :: RV
inputV = fromList [3, 2, 1]

params :: (RV, (MMP, MMP))
params =
  ( targetV,
    ( ((2 >< 3) [1, 2, 2, 3, 5, 6], fromList [0, 1]),
      ((3 >< 2) (repeat 8), fromList [0, 3, 0])
    )
  )

----------

exampleMini :: ParaLens' (Inp RV, MMP) () (Out RV)
exampleMini = argToPara .#. matMulLens -- . relu

exampleMiniLoss :: ParaLens' ((Inp RV, MMP), Tgt RV) () (Out R)
exampleMiniLoss = exampleMini .#. lossSmooth

exampleMini' :: ParaLRLens' ((Inp RV, MMP), Tgt RV) ()
exampleMini' = exampleMini .#. lossSmooth . lrSmooth 0.01

inputMini :: RV
inputMini = fromList [0, 1]

optimalParams :: MMP
optimalParams = ((2 >< 2) [1, 2, 3, 4], fromList [0, 1])

initialParams :: MMP
initialParams = ((2 >< 2) [-1, 1, -1, 1], fromList [0.5, 0.5])

runOptimal :: RV -> RV
runOptimal = runFullModel exampleMini . (,optimalParams)

genTargets :: [Inp RV] -> [(Inp RV, Tgt RV)]
genTargets = map (id &&& runOptimal)

targetMini :: RV
targetMini = runOptimal inputMini -- [2,5]

targets :: [(Inp RV, Tgt RV)]
targets = genTargets [fromList [a, b] | a <- [0 .. 2], b <- [1 .. 3]]

--

exampleMMini :: ParaLens' (Inp RV, RM) () (Out RV)
exampleMMini = argToPara .#. withGradDesc linear

exampleMMiniLoss :: ParaLens' ((Inp RV, RM), Tgt RV) () (Out R)
exampleMMiniLoss = exampleMMini .#. lossSmooth

exampleMMini' :: ParaLRLens' ((Inp RV, RM), Tgt RV) ()
exampleMMini' = exampleMMini .#. lossSmooth . lrSmooth 0.01

optimalParamsM :: RM
optimalParamsM = (1 >< 1) [0.5]

initParamsM :: RM
initParamsM = (1 >< 1) [-0.5]

inputMMini :: RV
inputMMini = fromList [2]

targetMMini :: RV
targetMMini = runFullModel exampleMMini (inputMMini, optimalParamsM) -- [0.5]

runOptimalM :: RV -> RV
runOptimalM = runFullModel exampleMMini . (,optimalParamsM)

genTargetsM :: [Inp RV] -> [(Inp RV, Tgt RV)]
genTargetsM = map (id &&& runOptimalM)

targetsM :: [(Inp RV, Tgt RV)]
targetsM = genTargetsM $ map (fromList . pure) [0 .. 3]

--

exampleMMiniB :: ParaLens' (Inp RM, RM) () (Out RM)
exampleMMiniB = argToPara .#. withGradDesc linear

-- exampleMMiniLossB :: ParaLens' ((Inp RV, RM), Tgt RV) () (Out R)
-- exampleMMiniLossB = exampleMMiniB .#. lossSmooth

-- exampleMMiniB' :: ParaLRLens' ((Inp RV, RM), Tgt RV) ()
-- exampleMMiniB' = exampleMMiniB .#. lossSmooth . lrSmooth 0.01

-- optimalParamsMB :: RM
-- optimalParamsMB = (1 >< 1) [0.5]

-- initParamsMB :: RM
-- initParamsMB = (1 >< 1) [-0.5]

-- inputMMiniB :: RV
-- inputMMiniB = fromList [2]

-- targetMMiniB :: RV
-- targetMMiniB = runFullModel exampleMMiniB (inputMMiniB, optimalParamsMB) -- [0.5]

-- runOptimalMB :: RV -> RV
-- runOptimalMB = runFullModel exampleMMiniB . (,optimalParamsMB)

-- genTargetsMB :: [Inp RV] -> [(Inp RV, Tgt RV)]
-- genTargetsMB = map (id &&& runOptimalMB)

-- targetsMB :: [(Inp RV, Tgt RV)]
-- targetsMB = genTargetsMB $ map (fromList . pure) [0 .. 3]