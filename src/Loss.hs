{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Loss where

import Control.Applicative
import Control.Arrow
import Control.Lens
import Control.Monad
import Core
import Data.Bits (xor)
import Data.Tuple (swap)
import qualified Data.Vector.Storable as VS
import Numeric.LinearAlgebra
import Types
import Sample

-------------------------
-- LOSS MAP --
-------------------------

-- loss1d :: Num n => ParaLens n n n n n n
-- loss1d = lens (uncurry (-)) rev
--   where

-- Quadratic Error [note: takes in negative alpha values]
--                     tgt vec  d(tgt)  guess vec d(pred) err alpha
lossSmooth :: ParaLens RVector RVector RVector RVector R R
lossSmooth = lens fwd (flip rev')
  where
    --      tgt vec  guess vec   err
    fwd :: (RVector, RVector) -> R -- (bt, bp)
    fwd = (* 0.5) . join dot . uncurry (-)

    --      alpha tgt vec  guess vec    d(tgt)   d(pred)
    rev' :: R -> (RVector, RVector) -> (RVector, RVector)
    -- This isn't what they said in the paper (swapped id and negate) but I think they're wrong
    rev' alpha (tgt, guess) = (d, -d)
      where d = scale alpha $ tgt - guess
    -- rev' alpha = (id &&& negate) . scale alpha . uncurry (-)

lossPoly :: ParaLens' ZVector ZVector ZVector -- in POLY_Z2
lossPoly = lens fwd rev
  where
    fwd :: (ZVector, ZVector) -> ZVector -- (bt, bp)
    fwd = uncurry $ VS.zipWith xor

    rev :: (ZVector, ZVector) -> ZVector -> (ZVector, ZVector)
    rev = const $ join (,)

softMax :: RVector -> RVector
-- softMax r = cmap (divSum r') r' where r' = cmap exp r
softMax = (divSum >>= cmap) . cmap exp
  where
    divSum :: RVector -> R -> R
    divSum = flip (/) . VS.sum

splitScale :: (Linear t a, Linear t b) => (a t, b t) -> t -> (a t, b t)
splitScale = flip $ liftA2 (***) scale scale

softMaxCELoss :: ParaLens' RVector RVector R
softMaxCELoss = lens fwd rev
  where
    fwd :: (RVector, RVector) -> R -- (bt, bp)
    fwd (bt, bp) = dot bt $ VS.zipWith f bp $ softMax bp
      where
        f :: R -> R -> R
        f bpi = (bpi -) . log

    rev :: (RVector, RVector) -> R -> (RVector, RVector)
    rev (bt, bp) = splitScale (-log q, q - bt)
      where
        q = softMax bp

deepDreamLoss :: ParaLens' RVector RVector R
deepDreamLoss = lens fwd rev
  where
    fwd :: (RVector, RVector) -> R -- (bt, bp)
    fwd = uncurry dot

    rev :: (RVector, RVector) -> R -> (RVector, RVector)
    rev = splitScale . swap