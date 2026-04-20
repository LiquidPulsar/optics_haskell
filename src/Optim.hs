{-# LANGUAGE RankNTypes #-}

module Optim where

import Control.Lens
import Core

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

lrPoly :: LRLens l l -- actually Z Z
lrPoly = learningRate id

-------------------------
-- OPTIMISERS --
-------------------------

gradUpdate :: (Num p) => Lens' p p
gradUpdate = lens id (+)

withGradDesc :: (Num p) => ParaLens p p a a' b b' -> ParaLens p p a a' b b'
withGradDesc = repara gradUpdate

-- gradUpdate' :: forall p p' . (Integral p, Num p, Integral p', Num p') => Lens p p p p'
-- gradUpdate' = lens id genPlus

-- genPlus :: (Integral p, Num p, Integral p', Num p', Integral p'', Num p'') => p -> p' -> p''
-- genPlus a b = fromIntegral $ a + fromIntegral b

-- gradUpdateMatrix :: Lens' RMatrix RMatrix
-- gradUpdateMatrix = gradUpdate