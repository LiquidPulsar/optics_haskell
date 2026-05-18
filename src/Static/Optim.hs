{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}

module Static.Optim where

import Control.Lens
import Core
import qualified Torch.Typed as T

type LRLens l l' = Lens l l' () ()

type LRLens' l = LRLens l l

type ParaLRLens p p' l l' = LRLens (p, l) (p', l')

type ParaLRLens' p l = ParaLRLens p p l l

-------------------------
-- LEARNING RATES --
-------------------------

learningRate :: (l -> l') -> LRLens l l'
learningRate alpha = lens (const ()) (const . alpha)

lrSmooth :: (Num l') => l' -> LRLens l l'
lrSmooth = learningRate . const . negate

scalarT :: forall dv dt a. (T.KnownDType dt, T.KnownDevice dv, T.Scalar a) => a -> T.Tensor dv dt '[]
scalarT = T.full @'[] @dt @dv

lrSmoothT ::
  forall dv dt a l.
  (T.KnownDType dt, T.KnownDevice dv, T.Scalar a) =>
  a -> LRLens l (T.Tensor dv dt '[])
lrSmoothT = lrSmooth . scalarT

-- lrPoly :: LRLens l l -- actually Z Z
-- lrPoly = learningRate id

-- -------------------------
-- -- OPTIMISERS --
-- -------------------------

gradUpdate :: (Num p) => Lens' p p -- Tensor impls Num so we're fine
gradUpdate = lens id (+)

withGradDesc :: (Num p) => ParaLens p p a a' b b' -> ParaLens p p a a' b b'
withGradDesc = repara gradUpdate

momrev :: (t ~ T.Tensor dv dt, T.Scalar a, Num a, T.KnownDevice dv) => a -> (t shape, t shape) -> t shape -> (t shape, t shape) 
momrev gamma (v, p) p' = (v', p + v')
  where v' = T.mulScalar (-gamma) v + p'

type Momentum = forall t dv dt shape. (t ~ T.Tensor dv dt, T.KnownDevice dv, T.StandardFloatingPointDTypeValidation dv dt) => Lens' (t shape, t shape) (t shape)
type Momentum2 = forall t dv dt shape. (t ~ T.Tensor dv dt, T.KnownDevice dv, T.StandardFloatingPointDTypeValidation dv dt) => Lens' ((t shape, t shape), t shape) (t shape)

momentum :: (T.Scalar a, Num a) => a -> Momentum
momentum = lens snd . momrev

nesterov :: (T.Scalar a, Num a) => a -> Momentum
nesterov gamma = lens (uncurry fwd) (momrev gamma)
  where
    -- fwd (v, p) = p + scale gamma v
    fwd = (+) . T.mulScalar gamma

adaGrad :: forall a . (T.Scalar a, Fractional a) => a -> Momentum
adaGrad eps = lens snd rev
  where
    delta :: a
    delta = 1e-7

    rev :: (t ~ T.Tensor dv dt, T.KnownDevice dv, T.StandardFloatingPointDTypeValidation dv dt) => (t shape, t shape) -> t shape -> (t shape, t shape)
    rev (g, p) p' = (g', p + update * p')
      where
        g' = g + p' * p'
        update = T.mulScalar eps . T.reciprocal . T.addScalar delta $ T.sqrt g'

-- -- Note: paper mentions a corrected estimate tracking time?
adam :: forall a . (T.Scalar a, Fractional a) => a -> a -> a -> Momentum2
adam β1 β2 ε = lens snd rev
  where
    delta :: a
    delta = 1e-8

    -- m: exp decaying avg of past grads
    -- v: exp decaying avg of past sq grads
    -- rev :: ((t R, t R), t R) -> t R -> ((t R, t R), t R)
    rev ((m, v), p) p' = ((m', v'), p + T.mulScalar ε update)
      where
        m'     = T.mulScalar β1 m + T.mulScalar (1 - β1) p'
        v'     = T.mulScalar β2 v + T.mulScalar (1 - β2) (p' * p')
        update = m' / T.addScalar delta (T.sqrt v')