{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TupleSections #-}
{- HLINT ignore "Redundant $" -}

module Core where

import Control.Lens
import Control.Arrow
import GHC.Exts

-- If C is a strict symmetric monoidal category (with monoidal product ⊗ and monoidal unit 𝐼) then we define a category Para(C) with

-- Lens A A' B B'
-- type Lens s t a b = forall f. Functor f => (a -> f b) -> s -> f t
-- lens :: (s -> a) -> (s -> b -> t) -> Lens s t a b
-- lens sa sbt afb s = sbt s <$> afb (sa s)

{-
This version uses an "unconsumed parameters" type sneaking alongside
This has the benefit of simple composition with (.) but will probs not
play as nice with the lens library
-}
-- type ParaLens p p' a a' b b' = forall q q' . Lens (p,a,q) (p',a',q') (b,q) (b',q')

type ParaLens p p' a a' b b' = Lens (p,a) (p',a') b b'
type ParaLens' p a b = ParaLens p p a a b b

-- ParaLens p p' a a' b b
-- Lens (p,a) (p',a') b b'
-- f :: (p,a) -> b
-- f* :: (p,a) -> b' -> (p',a')

type ParaIso p p' a a' b b' = Iso (p,a) (p',a') b b'
type ParaIso' p a b = ParaIso p p a a b b

{-
alongside gives the monoidal product
idLens is the unit
swapped is the symmetry
-}

--                        Lens ((),a) ((),a') b b'
toPara :: Lens a a' b b' -> ParaLens () () a a' b b'
toPara = (leftUnit .)
{-# INLINE toPara #-}

leftUnit :: ParaIso () () a a' a a'
leftUnit = iso snd ((),)
{-# INLINE leftUnit #-}

rightUnit :: ParaIso a a' () () a a'
rightUnit = iso fst (,())
{-# INLINE rightUnit #-}

-- idLens :: Iso a b a b
-- -- idLens = lens id (const id) -- or `curry snd` to match the paper def
-- idLens = id

rightLens :: Lens p p' q q' -> Lens (a, p) (b, p') (a, q) (b, q')
rightLens = inline alongside $ id -- somehow this $ is relevant
{-# INLINE rightLens #-}

leftLens :: Lens p p' q q' -> Lens (p, a) (p', b) (q, a) (q', b)
leftLens = flip (inline alongside) id
-- leftLens l = swapped . rightLens l . swapped
-- leftLens = bimap swapped swapped rightLens ?
{-# INLINE leftLens #-}

repara :: Lens q q' p p' -> ParaLens p p' a a' b b' -> ParaLens q q' a a' b b'
repara q = (leftLens q .)
{-# INLINE repara #-}

argToPara :: ParaIso p p' () () p p'
argToPara = rightUnit
{-# INLINE argToPara #-}

-------------------------
-- COMPOSITION --
-------------------------

-- https://hackage-content.haskell.org/package/lens-5.3.5/docs/Control-Lens-Type.html#t:LensLike
-- "Since every Iso is both a valid Lens and a valid Prism,"
swapFst :: Iso ((a,b),c) ((a',b'),c') ((b,a),c) ((b',a'),c')
swapFst = alongsideIso swapped id
{-# INLINE swapFst #-}

-- TODO: any better?
alongsideIso :: Iso a b c d -> Iso a' b' c' d' -> Iso (a,a') (b,b') (c,c') (d,d')
alongsideIso i i' = iso (ac *** ac') (db *** db')
  where
    (ac, db) = withIso i (,)
    (ac', db') = withIso i' (,)
{-# INLINE alongsideIso #-}

rotate :: Iso ((a,b),c) ((a',b'),c') (a,(b,c)) (a',(b',c'))
rotate = iso fwd rev
  where
    fwd ((a,b),c) = (a,(b,c))
    rev (a,(b,c)) = ((a,b),c)
{-# INLINE rotate #-}


-- TODO: better name?

infixr 8 .#. -- one less than (.) so that we can do things like: "a .#. b . c .#. d"
--       Lens (p,a) (p',a') b b' -> Lens (q,b) (q',b') c c' -> Lens ((p,q),a) ((p',q'),a') c c'
(.#.) :: ParaLens p p' a a' b b' -> ParaLens q q' b b' c c' -> ParaLens (p,q) (p',q') a a' c c'
(.#.) ab bc = swapFst . rotate . rightLens ab . bc
{-# INLINE (.#.) #-} -- actually useful
-- (.#.) ab = (z .)
--   where
--     x :: Iso ((p,q),a) ((p',q'),a') (q,(p,a)) (q',(p',a'))
--     x = swapFst . rotate

--     y :: Lens (q,(p,a)) (q',(p',a')) (q,b) (q',b')
--     y = rightLens ab

--     z :: Lens ((p,q),a) ((p',q'),a') (q,b) (q',b')
--     z = x . y

liftUpdate :: Lens' p p -> Lens' [p] [p]
liftUpdate ul = lens (map $ view ul) $ flip (zipWith (set ul))
-- liftUpdate ul = lens (map $ view ul) (\ps gs -> zipWith (set ul) gs ps)
{-# INLINE liftUpdate #-}