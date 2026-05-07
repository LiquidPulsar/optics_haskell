{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

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

scalarT :: forall device dtype a. (T.KnownDType dtype, T.KnownDevice device, T.Scalar a) => a -> T.Tensor device dtype '[]
scalarT = T.full @'[] @dtype @device

lrSmoothT ::
  forall device dtype a l.
  (T.KnownDType dtype, T.KnownDevice device, T.Scalar a) =>
  a -> LRLens l (T.Tensor device dtype '[])
lrSmoothT = lrSmooth . scalarT

-- lrPoly :: LRLens l l -- actually Z Z
-- lrPoly = learningRate id

-- -------------------------
-- -- OPTIMISERS --
-- -------------------------

gradUpdate :: (Num p) => Lens' p p
gradUpdate = lens id (+)

withGradDesc :: (Num p) => ParaLens p p a a' b b' -> ParaLens p p a a' b b'
withGradDesc = repara gradUpdate

-- -- Could drop the Num p instance at some perf cost if we use an extra negate call after using Num (t p)
-- momrev :: (Linear p t, Num p, Num (t p)) => p -> (t p, t p) -> t p -> (t p, t p)
-- momrev gamma (v, p) p' = (v', p + v')
--   where v' = scale (-gamma) v + p'

-- type Momentum p = forall t. (Num p, Num (t p), Linear p t, Container t p, Floating (t R)) => Lens' (t p, t p) (t p)

-- momentum :: forall p t. (Num p, Num (t p), Linear p t) => p -> Lens' (t p, t p) (t p)
-- momentum = lens snd . momrev

-- nesterov :: forall p t. (Num p, Num (t p), Linear p t) => p -> Lens' (t p, t p) (t p)
-- nesterov gamma = lens (uncurry fwd) (momrev gamma)
--   where
--     -- fwd (v, p) = p + scale gamma v
--     fwd = (+) . scale gamma

-- adaGrad :: R -> Momentum R
-- adaGrad eps = lens snd rev
--   where
--     delta :: R
--     delta = 1e-7

--     -- rev :: (t R, t R) -> t R -> (t R, t R)
--     rev (g, p) p' = (g', p + update * p')
--       where
--         g' = g + p * p'
--         update = scale eps . recip . cmap (delta +) $ sqrt g' -- yeah I don't like hmatrix, why is "addConstant" internal aaaaaa

-- -- Note: paper mentions a corrected estimate tracking time?
-- adam :: forall t. (Num (t R), Linear R t, Container t R, Floating (t R))
--           => R -> R -> R -> Lens' ((t R, t R), t R) (t R)
-- adam β1 β2 ε = lens snd rev
--   where
--     delta :: R
--     delta = 1e-8

--     -- m: exp decaying avg of past grads
--     -- v: exp decaying avg of past sq grads
--     rev :: ((t R, t R), t R) -> t R -> ((t R, t R), t R)
--     rev ((m, v), p) p' = ((m', v'), p + scale ε update)
--       where
--         m'     = scale β1 m + scale (1 - β1) p'
--         v'     = scale β2 v + scale (1 - β2) (p' * p')
--         update = m' / cmap ((delta +) . sqrt) v'