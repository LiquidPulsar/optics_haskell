{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE FlexibleContexts #-}

module Static.Loss where

import Control.Lens
import Core
import Torch (numel)
import qualified Torch.Typed as T
import GHC.TypeLits

-------------------------
-- LOSS MAP --
-------------------------

-- loss1d :: Num n => ParaLens n n n n n n
-- loss1d = lens (uncurry (-)) rev
--   where

-- Quadratic Error [note: takes in negative alpha values]
--                     tgt vec  d(tgt)  guess vec d(pred) err alpha

type TrivialFacts shape = ( -- See below impl of lossSmooth
    shape ~ T.Broadcast shape shape, -- ... yeah blame the T.sub :/
    shape ~ T.Reverse (T.Reverse shape) -- blame the mul)
  )

lossSmooth ::
  forall t shape dv dt.
  ( t ~ T.Tensor dv dt,
    TrivialFacts shape,
    T.BasicArithmeticDTypeIsValid dv dt,
    T.StandardFloatingPointDTypeValidation dv dt
  ) =>
  ParaLens' (t shape) (t shape) (t '[]) -- r stays on dv, extract with T.asValue
lossSmooth = lens fwd (flip rev')
  where
    --      tgt vec  guess vec   err
    fwd :: (t shape, t shape) -> t '[] -- (bt, bp)
    fwd = uncurry $ T.mseLoss @T.ReduceMean

    --      alpha tgt vec  guess vec    d(tgt)   d(pred)
    rev' :: t '[] -> (t shape, t shape) -> (t shape, t shape)
    -- This isn't what they said in the paper (swapped id and negate) but I think they're wrong
    -- Also: include the 2/N factor from differentiating mean((tgt-guess)^2)
    rev' alpha (tgt, guess) = (d, T.neg d)
      where
        n = fromIntegral (numel (T.toDynamic tgt)) :: Float
        d = T.mulScalar (2 / n) . T.mul alpha $ T.sub tgt guess

-- lossPoly :: ParaLens' ZVector ZVector ZVector -- in POLY_Z2
-- lossPoly = lens fwd rev
--   where
--     fwd :: (ZVector, ZVector) -> ZVector -- (bt, bp)
--     fwd = uncurry $ VS.zipWith xor

--     rev :: (ZVector, ZVector) -> ZVector -> (ZVector, ZVector)
--     rev = const $ join (,)

softMaxCELoss ::
  forall t x c dv dt.
  ( t ~ T.Tensor dv dt,
    T.All KnownNat '[x, c],
    T.AllDimsPositive '[x],
    T.BasicArithmeticDTypeIsValid dv dt,
    T.StandardFloatingPointDTypeValidation dv dt,
    T.SumDType dt ~ dt, 
    T.KnownDType dt, 
    T.KnownDevice dv, 
    T.SumDTypeIsValid dv dt, 
    T.MeanDTypeValidation dv dt
  ) =>
  ParaLens' (t '[x, c]) (t '[x, c]) (t '[])
softMaxCELoss = lens fwd rev
  where
    fwd :: (t '[x, c], t '[x, c]) -> t '[]
    fwd (bt, bp) = negate . T.meanAll . T.sumDim @1 $ bt * T.logSoftmax @1 bp

    rev :: (t '[x, c], t '[x, c]) -> t '[] -> (t '[x, c], t '[x, c])
    rev (bt, bp) d = (T.mul d $ negate $ T.log q, T.mul d $ q - bt)
      where
        q = T.softmax @1 bp   -- '[x,c], sums to 1 over class dim

deepDreamLoss ::
  forall t shape dv dt.
  ( t ~ T.Tensor dv dt,
    T.KnownShape shape,
    T.BasicArithmeticDTypeIsValid dv dt,
    T.StandardFloatingPointDTypeValidation dv dt, 
    T.SumDTypeIsValid dv dt,
    T.SumDType dt ~ dt,
    T.Reverse shape ~ shape
  ) =>
  ParaLens' (t shape) (t shape) (t '[])
deepDreamLoss = lens fwd rev
  where
    fwd :: (t shape, t shape) -> t '[]
    fwd (bt, bp) = T.sumAll $ T.mul bt bp

    rev :: (t shape, t shape) -> t '[] -> (t shape, t shape)
    rev (bt, bp) g = (T.mul bp g, T.mul bt g)