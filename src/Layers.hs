{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Layers where

import Control.Lens
import Control.Monad
import Core
import Numeric.LinearAlgebra
import Optim
import Sample
import Types
import Control.Arrow

{-
Goal	                        Use
Apply R -> R to vector	        cmap
Apply R -> R -> R	            zipWith
Create vector from Int -> R	    build
Scale vector	                scale
Square elements	                v * v
-}

-------------------------
-- CARTESIAN REVERSE DIFFERENTIAL CATEGORIES --
-------------------------

-------------------------
-- LAYERS --
-------------------------

--                                                        m           x     y
linear :: (Sample f, Floating e, Numeric e) => ParaLens' (Matrix e) (f e) (f e)
linear = lens (uncurry applyLinear) rev
  where
    rev (m, x) y = (accumGrad y x, tr m `applyLinear` y)

--                            m x y
addLens :: forall e f. (Sample f, ProdNum e) => ParaLens' (Vector e) (f e) (f e)
addLens = lens (uncurry addBias) rev
  where
    rev :: (Vector e, f e) -> f e -> (Vector e, f e)
    rev = const $ sumSamples &&& id

expit :: Floating a => a -> a
expit = recip . (1 +) . exp . negate

sigmoid :: forall e f. (Sample f, Floating e, Numeric e, Num (f e)) => Lens' (f e) (f e)
sigmoid = lens (cmap expit) rev
  where
    fwd :: f e -> f e
    fwd = cmap expit

    rev :: f e -> f e -> f e
    rev = (*) . ap (*) (1 -) . fwd

relu :: (Sample f, Floating e, Numeric e, Ord e, Num (f e)) => Lens' (f e) (f e)
relu = lens (cmap $ max 0) ((*) . step)

matMulLens :: (Sample f, Num (f R)) => ParaLens' MMP (f R) (f R)
matMulLens = withGradDesc linear .#. withGradDesc addLens