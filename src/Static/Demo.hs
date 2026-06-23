{-# LANGUAGE DataKinds #-}

module Static.Demo where

import qualified Torch.Typed as T
import qualified Static.Layers as L
import Static.Layers hiding (MMP)
import Core
import Models

type DV = '(T.CPU, 0)
type DT = T.Double

type Tensor s = T.Tensor DV DT s
type MMP o i = L.MMP DV DT o i

p = undefined
oops = undefined

foo :: ParaLens' (MMP 6 4) (Tensor [b,4]) (Tensor [b,6])
foo = matMulLens

-- x :: Tensor [1,6]
-- x = run (p :: MMP 6 4, oops :: Tensor [1,10])
--   where run = runModel foo