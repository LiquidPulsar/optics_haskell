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

example :: ParaLens' (Inp RVector, (MMP, MMP)) () (Out RVector)
example = argToPara .#. matMulLens . relu .#. matMulLens . relu

example' :: ParaLRLens' ((Inp RVector, (MMP, MMP)), Tgt RVector) ()
example' = example .#. lossSmooth . lrSmooth 1

targetV :: RVector
targetV = fromList [1, 2, 3]

inputV :: RVector
inputV = fromList [3, 2, 1]

params :: (RVector, (MMP, MMP))
params =
  ( targetV,
    ( ((2 >< 3) [1, 2, 2, 3, 5, 6], fromList [0, 1]),
      ((3 >< 2) (repeat 8), fromList [0, 3, 0])
    )
  )

----------

exampleMini :: ParaLens' (Inp RVector, MMP) () (Out RVector)
exampleMini = argToPara .#. matMulLens -- . relu

exampleMiniLoss :: ParaLens' ((Inp RVector, MMP), Tgt RVector) () (Out R)
exampleMiniLoss = exampleMini .#. lossSmooth

exampleMini' :: ParaLRLens' ((Inp RVector, MMP), Tgt RVector) ()
exampleMini' = exampleMini .#. lossSmooth . lrSmooth 0.01

inputMini :: RVector
inputMini = fromList [0, 1]

optimalParams :: MMP
optimalParams = ((2 >< 2) [1, 2, 3, 4], fromList [0, 1])

initialParams :: MMP
initialParams = ((2 >< 2) [-1, 1, -1, 1], fromList [0.5, 0.5])

runOptimal :: RVector -> RVector
runOptimal = runFullModel exampleMini . (,optimalParams)

genTargets :: [Inp RVector] -> [(Inp RVector, Tgt RVector)]
genTargets = map (id &&& runOptimal)

targetMini :: RVector
targetMini = runOptimal inputMini -- [2,5]

targets :: [(Inp RVector, Tgt RVector)]
targets = genTargets [fromList [a, b] | a <- [0 .. 2], b <- [1 .. 3]]

--

exampleMMini :: ParaLens' (Inp RVector, RMatrix) () (Out RVector)
exampleMMini = argToPara .#. withGradDesc linear

exampleMMiniLoss :: ParaLens' ((Inp RVector, RMatrix), Tgt RVector) () (Out R)
exampleMMiniLoss = exampleMMini .#. lossSmooth

exampleMMini' :: ParaLRLens' ((Inp RVector, RMatrix), Tgt RVector) ()
exampleMMini' = exampleMMini .#. lossSmooth . lrSmooth 0.01

optimalParamsM :: RMatrix
optimalParamsM = (1 >< 1) [0.5]

initParamsM :: RMatrix
initParamsM = (1 >< 1) [-0.5]

inputMMini :: RVector
inputMMini = fromList [2]

targetMMini :: RVector
targetMMini = runFullModel exampleMMini (inputMMini, optimalParamsM) -- [0.5]

runOptimalM :: RVector -> RVector
runOptimalM = runFullModel exampleMMini . (,optimalParamsM)

genTargetsM :: [Inp RVector] -> [(Inp RVector, Tgt RVector)]
genTargetsM = map (id &&& runOptimalM)

targetsM :: [(Inp RVector, Tgt RVector)]
targetsM = genTargetsM $ map (fromList . pure) [0 .. 3]

--

exampleMMiniB :: ParaLens' (Inp RMatrix, RMatrix) () (Out RMatrix)
exampleMMiniB = argToPara .#. withGradDesc linear

-- exampleMMiniLossB :: ParaLens' ((Inp RVector, RMatrix), Tgt RVector) () (Out R)
-- exampleMMiniLossB = exampleMMiniB .#. lossSmooth

-- exampleMMiniB' :: ParaLRLens' ((Inp RVector, RMatrix), Tgt RVector) ()
-- exampleMMiniB' = exampleMMiniB .#. lossSmooth . lrSmooth 0.01

-- optimalParamsMB :: RMatrix
-- optimalParamsMB = (1 >< 1) [0.5]

-- initParamsMB :: RMatrix
-- initParamsMB = (1 >< 1) [-0.5]

-- inputMMiniB :: RVector
-- inputMMiniB = fromList [2]

-- targetMMiniB :: RVector
-- targetMMiniB = runFullModel exampleMMiniB (inputMMiniB, optimalParamsMB) -- [0.5]

-- runOptimalMB :: RVector -> RVector
-- runOptimalMB = runFullModel exampleMMiniB . (,optimalParamsMB)

-- genTargetsMB :: [Inp RVector] -> [(Inp RVector, Tgt RVector)]
-- genTargetsMB = map (id &&& runOptimalMB)

-- targetsMB :: [(Inp RVector, Tgt RVector)]
-- targetsMB = genTargetsMB $ map (fromList . pure) [0 .. 3]