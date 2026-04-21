{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MonoLocalBinds #-}

module Layers where

import Control.Lens hiding ((<.>))
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

-------------------------
-- MATMUL & OPTIMISED  --
-------------------------

matMulLensCore :: (Sample f, Num (f R)) => ParaLens' MMP (f R) (f R)
matMulLensCore = linear .#. addLens

matMulLens :: (Sample f, Num (f R)) => ParaLens' MMP (f R) (f R)
matMulLens = repara r matMulLensCore
  where
    r :: Lens' MMP MMP
    r = alongside gradUpdate gradUpdate

withMomentum :: (Sample f, Num (f R)) => Momentum R -> ParaLens' (MMP, MMP) (f R) (f R)
withMomentum mom = repara r matMulLensCore
  where
    q :: Lens' ((RM, RM), (RV, RV)) MMP
    q = alongside mom mom

    r :: Lens' (MMP, MMP) MMP
    r = rotate' . q

rotate' :: Iso ((a,b),(c,d)) ((a',c'),(b',d')) ((a,c),(b,d)) ((a',b'),(c',d'))
rotate' = iso fwd rev
  where
    fwd ((a,b),(c,d)) = ((a,c),(b,d))
    rev ((a,c),(b,d)) = ((a,b),(c,d))

matMulLensMom :: (Sample f, Num (f R)) => R -> ParaLens' (MMP, MMP) (f R) (f R)
matMulLensMom gamma = withMomentum $ momentum gamma

matMulLensNest :: (Sample f, Num (f R)) => R -> ParaLens' (MMP, MMP) (f R) (f R)
matMulLensNest gamma = withMomentum $ nesterov gamma

matMulLensAda :: (Sample f, Num (f R)) => R -> ParaLens' (MMP, MMP) (f R) (f R)
matMulLensAda gamma = withMomentum $ adaGrad gamma

matMulLensAdam :: (Sample f, Num (f R)) => R -> R -> R -> ParaLens' (MMP, MMP, MMP) (f R) (f R)
matMulLensAdam b1 b2 eps = repara r matMulLensCore
  where
    ad :: forall t . (Num (t R), Linear R t, Container t R, Floating (t R)) => Lens' ((t R, t R), t R) (t R)
    ad = adam b1 b2 eps

    q :: Lens' (((RM, RM), RM), ((RV, RV), RV)) MMP
    q = alongside ad ad -- kinda neat that we still maintain the flexibility to spec to matrix or vector here!

    r :: Lens' (MMP, MMP, MMP) MMP
    r = rot . q

    -- lazy type def, actually more general but won't use it elsewhere anyway
    rot :: Iso' ((a,d),(b,e),(c,f)) (((a,b),c),((d,e),f))
    rot = iso fwd rev
      where
        fwd ((a,d),(b,e),(c,f)) = (((a,b),c),((d,e),f))
        rev (((a,b),c),((d,e),f)) = ((a,d),(b,e),(c,f))

-------------------------
-- CONVOLUTION --
-------------------------

rot180 :: Matrix Double -> Matrix Double
rot180 = fliprl . flipud

convolve2D :: ParaLens' (Matrix Double) (Matrix Double) (Matrix Double)
convolve2D = lens (uncurry corr2) rev
  where
    -- fwd :: (Matrix Double, Matrix Double) -> Matrix Double
    -- fwd (k, a) = corr2 k a

    rev :: (Matrix Double, Matrix Double) -> Matrix Double -> (Matrix Double, Matrix Double)
    rev (k, a) dy = (dk, da)
      where
        dk = corr2 dy a
        da = conv2 (rot180 k) dy

type Image = [Matrix Double]
type Kernels = [[Matrix Double]]        -- [out_channel][in_channel]

correlate2D :: ParaLens' Kernels Image Image
correlate2D = lens fwd rev
  where
    fwd (ks, img) =
        [ foldl1 add [ corr2 k i | (k, i) <- zip ks_o img ]
        | ks_o <- ks ]

    rev (ks, img) dy = (dks, dimg)
      where
        dks  = [ [ corr2 i d | i <- img ] | d <- dy ]
        dimg = [ foldl1 add [ conv2 (rot180 (ks_o !! ic)) d
                             | (ks_o, d) <- zip ks dy ]
               | ic <- [0..length img - 1] ]

maxPool2DChannel :: Int -> Int -> Lens' (Matrix Double) (Matrix Double)
maxPool2DChannel kh kw = lens fwd rev
  where
    blockIndices m = [(i, j) | i <- [0..rows m `div` kh - 1]
                              , j <- [0..cols m `div` kw - 1]]

    getBlock m i j = subMatrix (i*kh, j*kw) (kh, kw) m

    fwd :: Matrix Double -> Matrix Double
    fwd m = let oh = rows m `div` kh
                ow = cols m `div` kw
            in (oh><ow) [maxElement (getBlock m i j) | (i,j) <- blockIndices m]

    rev :: Matrix Double -> Matrix Double -> Matrix Double
    rev x dy =
        let updates = do
                (i, j) <- blockIndices x
                let block    = getBlock x i j
                    (_pi, pj) = maxIndex block -- shadows pi
                    grad     = dy `atIndex` (i, j)
                return ((i*kh + _pi, j*kw + pj), grad)
        in accum (konst 0 (rows x, cols x)) (+) updates


-- TODO: Smth possible with traverse here to lift the lens up
-- maxPool2D kh kw = traverse . maxPool2DChannel kh kw
maxPool2D :: Int -> Int -> Lens' Image Image
maxPool2D kh kw = lens fwd rev
  where
    cl :: Lens' (Matrix Double) (Matrix Double)
    cl = maxPool2DChannel kh kw
    fwd = map (view cl)
    rev xs dys = zipWith (set cl) dys xs -- TODO: is the order right?