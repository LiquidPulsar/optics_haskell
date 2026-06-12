{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeOperators #-}

module Compile where
import Static.Layers (matMulLens, CanMMLens)
import qualified Static.Layers as L
import Core
import qualified Torch.Typed as T
import Control.Lens
import Models (runModel)

-- import Control.Lens
-- import Core (leftLens)
-- import qualified Torch.Typed as T
-- import qualified Torch as U
-- import qualified Torch.Functional.Internal as I
-- import GHC.IO (unsafePerformIO)

-- import qualified Torch.Internal.Const as ATen

-- import qualified Torch.Internal.Managed.Cast

-- import qualified Torch.Internal.Managed.Native as ATen
-- import qualified Torch.Internal.Managed.Type.Scalar as ATen
-- import qualified Torch.Internal.Managed.Type.Tensor as ATen
-- import qualified Torch.Internal.Managed.Type.Tuple as ATen
-- import qualified Torch.Internal.Type as ATen
-- import Torch.Internal.Cast (cast2)


-- basic :: (Num a) => Lens' a a
-- basic = lens (+ 1) (+)
-- {-# INLINE basic #-}

-- foo :: Int -> Int
-- foo = foo' -- this inlines to +1 correctly

-- foo' :: (Num a) => a -> a
-- foo' = view basic

-- med :: (Num a) => Lens' (a, b) (a, b)
-- med = leftLens basic
-- {-# INLINE med #-}
-- {-# SPECIALIZE med :: Lens' (Int, b) (Int, b) #-}

-- med' :: (Int, a) -> (Int, a)
-- med' = view med -- This reduces into med'' :D !!
-- {-# INLINE med' #-}

-- med'' :: (Int, a) -> (Int, a)
-- med'' (x, y) = (x + 1, y)

------

-- {-# RULES
-- "i @ m.T + b -> linear"   forall m i b. T.add (T.matmul i (T.transpose @0 @1 m)) b = T.linear m i b
-- "i @ m.T + b -> linear_2" forall m i b. U.add (U.matmul i (U.transpose 0 1 m)) b = I.linear m i b
-- #-}

-- {-# RULES 
-- "foo" forall a b. unsafePerformIO $ cast2 ATen.matmul_tt a b = unsafePerformIO $ cast2 ATen.matmul_tt a b
-- #-}

-- x a b = unsafePerformIO $ cast2 ATen.matmul_tt a b

-- rulepls m i b = T.add (T.matmul i (T.transpose @0 @1 m)) b

-- {-# RULES
-- "i @ m.T + b -> linear" forall m i b. T.add (T.matmul i (T.transpose @0 @1 m)) b = T.linear m i b
-- #-}





-- type DV = '(T.CPU, 0)
-- type DT = T.Double

-- p = undefined
-- oops = undefined

-- type Tensor s = T.Tensor DV DT s
-- type MMP o i = L.MMP DV DT o i

-- foo :: ParaLens' (MMP 6 4) (Tensor [b, 4]) (Tensor [b, 6])
-- foo = matMulLens

-- x :: Tensor [1, 6]
-- x = run (p :: MMP 6 4, oops :: Tensor [1, 10])
--   where run = runModel foo