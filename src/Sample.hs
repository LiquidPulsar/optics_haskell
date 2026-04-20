{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE RankNTypes #-}

module Sample where

import Numeric.LinearAlgebra
import Prelude hiding ((<>))

type ProdNum e = (Product e, Numeric e)

class
  ( forall e. (ProdNum e) => Container f e,
    forall e. (ProdNum e) => Additive (f e)
  ) =>
  Sample f
  where
  applyLinear :: (ProdNum e) => Matrix e -> f e -> f e
  accumGrad :: (ProdNum e) => f e -> f e -> Matrix e
  sumSamples :: (ProdNum e) => f e -> Vector e
  addBias :: (ProdNum e) => Vector e -> f e -> f e -- broadcast

instance Sample Vector where
  applyLinear = (#>)
  accumGrad = outer
  sumSamples = id
  addBias = add

sumRows :: (ProdNum e) => Matrix e -> Vector e
sumRows m = m #> konst 1 (cols m)

instance Sample Matrix where -- columns = samples
  applyLinear = (<>)
  accumGrad y x = y <> tr x
  sumSamples = sumRows
  addBias = add . asColumn