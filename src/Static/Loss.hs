{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ConstraintKinds #-}

module Static.Loss where

import Control.Arrow
import Control.Lens
import Core
import qualified Torch.Typed as T

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
  forall t shape r rv device dtype.
  ( t ~ T.Tensor device dtype,
    rv ~ t shape,
    r ~ t '[],
    TrivialFacts shape,
    T.BasicArithmeticDTypeIsValid device dtype,
    T.StandardFloatingPointDTypeValidation device dtype
  ) =>
  ParaLens rv rv rv rv r r -- r stays on device, extract with T.asValue
lossSmooth = lens fwd (flip rev')
  where
    --      tgt vec  guess vec   err
    fwd :: (rv, rv) -> r -- (bt, bp)
    fwd = uncurry $ T.mseLoss @T.ReduceMean

    --      alpha tgt vec  guess vec    d(tgt)   d(pred)
    rev' :: r -> (rv, rv) -> (rv, rv)
    -- This isn't what they said in the paper (swapped id and negate) but I think they're wrong
    rev' alpha = (id &&& T.neg) . T.mul alpha . uncurry T.sub
    -- rev' alpha (tgt, guess) = (d, T.neg d)
    --   where d = T.mul alpha $ T.sub tgt guess

-- lossPoly :: ParaLens' ZVector ZVector ZVector -- in POLY_Z2
-- lossPoly = lens fwd rev
--   where
--     fwd :: (ZVector, ZVector) -> ZVector -- (bt, bp)
--     fwd = uncurry $ VS.zipWith xor

--     rev :: (ZVector, ZVector) -> ZVector -> (ZVector, ZVector)
--     rev = const $ join (,)

-- softMax :: RV -> RV
-- -- softMax r = cmap (divSum r') r' where r' = cmap exp r
-- softMax = (divSum >>= cmap) . cmap exp
--   where
--     divSum :: RV -> R -> R
--     divSum = flip (/) . VS.sum

-- splitScale :: (Linear t a, Linear t b) => (a t, b t) -> t -> (a t, b t)
-- splitScale = flip $ liftA2 (***) scale scale

-- softMaxCELoss :: ParaLens' RV RV R
-- softMaxCELoss = lens fwd rev
--   where
--     fwd :: (RV, RV) -> R -- (bt, bp)
--     fwd (bt, bp) = dot bt $ VS.zipWith f bp $ softMax bp
--       where
--         f :: R -> R -> R
--         f bpi = (bpi -) . log

--     rev :: (RV, RV) -> R -> (RV, RV)
--     rev (bt, bp) = splitScale (-log q, q - bt)
--       where
--         q = softMax bp

-- deepDreamLoss :: ParaLens' RV RV R
-- deepDreamLoss = lens fwd rev
--   where
--     fwd :: (RV, RV) -> R -- (bt, bp)
--     fwd = uncurry dot

--     rev :: (RV, RV) -> R -> (RV, RV)
--     rev = splitScale . swap