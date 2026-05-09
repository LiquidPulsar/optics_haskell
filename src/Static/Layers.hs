{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
-- {-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Static.Layers where

import Control.Lens
import Control.Monad
import Core
import Data.Proxy
import GHC.TypeNats
import Optim
import qualified Torch as U
import Torch.Typed (Init, Tensor (UnsafeMkTensor), toDynamic, type (++))
import qualified Torch.Typed as T
import Control.Arrow

-------------------------
-- CARTESIAN REVERSE DIFFERENTIAL CATEGORIES --
-------------------------

-------------------------
-- LAYERS --
-------------------------

transp :: Tensor device dtype shape -> Tensor device dtype (T.SetValue (T.SetValue shape 0 (T.GetValue shape 1)) 1 (T.GetValue shape 0))
transp = T.transpose @0 @1

linear ::
  forall batch i o device dtype.
  (
    T.MatMulDTypeIsValid device dtype
  ) =>
  ParaLens'
    (Tensor device dtype '[o, i])
    (Tensor device dtype '[batch, i])
    (Tensor device dtype '[batch, o])
linear = lens fwd rev
  where
    fwd (w, x) = T.matmul x $ transp w

    rev (w, x) grad = (dW, dx)
      where
        dW = T.matmul (transp grad) x
        dx = T.matmul grad w
{-# INLINE linear #-}

linear' ::
  forall t i o shape device dtype.
  ( t ~ Tensor device dtype,
    T.IsSuffixOf '[i] shape,
    KnownNat i,
    KnownNat o
    --  , T.KnownShape shape
  ) =>
  ParaLens'
    (t '[o, i])
    (t shape)
    (t (Init shape ++ '[o]))
linear' = lens fwd rev
  where
    fwd :: (t '[o, i], t shape) -> t (Init shape ++ '[o])
    fwd (w, x) =
      UnsafeMkTensor $
        -- [..., i] @ [i, o] -> [..., o]
        U.matmul (toDynamic x) (toDynamic $ T.transpose @0 @1 w)

    rev ::
      (t '[o, i], t shape) ->
      t (Init shape ++ '[o]) ->
      (t '[o, i], t shape)
    rev (w, x) grad = (dW, dx)
      where
        i = fromIntegral $ natVal (Proxy @i)
        o = fromIntegral $ natVal (Proxy @o)

        -- reshape to [batch, o] and [batch, i], then grad^T @ x -> [o, i]
        dW :: t '[o, i]
        dW =
          UnsafeMkTensor $
            let g = U.reshape [-1, o] (toDynamic grad)
                x' = U.reshape [-1, i] (toDynamic x)
             in U.matmul (U.transpose2D g) x'

        -- [..., o] @ [o, i] -> [..., i]
        dx :: t shape
        dx = UnsafeMkTensor $ U.matmul (toDynamic grad) (toDynamic w)

type CanAddLens device dtype =
  ( 
    T.BasicArithmeticDTypeIsValid device dtype,
    T.SumDTypeIsValid device dtype,
    T.SumDType dtype ~ dtype
  )

addLens ::
  forall shape b device dtype t.
  ( t ~ Tensor device dtype,
    CanAddLens device dtype,
    b:shape ~ T.Broadcast shape (b:shape) -- trivial tbh
  ) =>
  ParaLens' (t shape) (t (b:shape)) (t (b:shape))
addLens = lens fwd rev
  where
    fwd :: (t shape, t (b:shape)) -> t (b:shape)
    fwd = uncurry T.add
    rev :: (t shape, t (b:shape)) -> t (b:shape) -> (t shape, t (b:shape))
    rev _ = T.sumDim @0 &&& id
{-# INLINE addLens #-}

-- Note: Can be better with fully known shapes
addLens' ::
  forall shape shape' shape'' device dtype t.
  ( t ~ Tensor device dtype,
    CanAddLens device dtype,
    T.KnownShape shape,
    shape'' ~ T.Broadcast shape shape',
    shape'' ~ shape',
    shape `T.IsSuffixOf` shape'
  ) =>
  ParaLens' (t shape) (t shape') (t shape'')
addLens' = lens fwd rev
  where
    fwd :: (t shape, t shape') -> t shape''
    fwd (b, x) = T.add b x
    rev :: (t shape, t shape') -> t shape'' -> (t shape, t shape')
    rev _ x' = (x''', x')
      where
        -- Safe as shape is suffix of shape' so multiples will work
        x'' :: t (w : shape) -- There exists a w, doesn't really matter what it is
        x'' = UnsafeMkTensor $ U.reshape (-1 : T.shapeVal @shape) $ toDynamic x'

        x''' :: t shape
        x''' = T.sumDim @0 x''
{-# INLINABLE addLens' #-}

sigmoid ::
  forall shape device dtype t.
  ( t ~ Tensor device dtype,
    T.StandardFloatingPointDTypeValidation device dtype,
    T.KnownDevice device
  ) =>
  Lens' (t shape) (t shape)
sigmoid = lens T.sigmoid rev
  where
    rev :: t shape -> t shape -> t shape
    rev = (*) . ap (*) (1 -) . T.sigmoid
{-# INLINE sigmoid #-}

relu ::
  forall shape device dtype t.
  ( t ~ Tensor device dtype,
    shape ~ T.Broadcast shape shape,
    T.StandardFloatingPointDTypeValidation device dtype,
    T.KnownDevice device,
    T.ComparisonDTypeIsValid device dtype,
    T.KnownDType dtype
  ) =>
  Lens' (t shape) (t shape)
relu = lens T.relu rev
  where
    rev :: t shape -> t shape -> t shape
    rev = (*) . heaviside
{-# INLINE relu #-}

heaviside ::
  forall shape device dtype t.
  ( t ~ Tensor device dtype,
    shape ~ T.Broadcast shape shape,
    T.StandardFloatingPointDTypeValidation device dtype,
    T.ComparisonDTypeIsValid device dtype,
    T.KnownDType dtype
  ) =>
  t shape ->
  t shape
-- TODO: pretty sure there is better using geScalar...
heaviside = T.toDType @dtype @T.Bool . liftA2 ($) T.gt T.zerosLike

-------------------------
-- MATMUL & OPTIMISED  --
-------------------------

type CanMMLens device dtype = ( T.MatMulDTypeIsValid device dtype, CanAddLens device dtype)

matMulLensCore ::
  forall t batch i o device dtype.
  ( t ~ Tensor device dtype,
    CanMMLens device dtype
  ) =>
  ParaLens' (t '[o, i], t '[o]) (t '[batch, i]) (t '[batch, o])
matMulLensCore = linear .#. addLens
{-# INLINE matMulLensCore #-}

type MMP device dtype o i = (Tensor device dtype '[o, i], Tensor device dtype '[o])

matMulLens ::
  forall t mmp batch i o device dtype.
  ( t ~ Tensor device dtype,
    mmp ~ MMP device dtype o i,
    CanMMLens device dtype,
    T.KnownDevice device
  ) =>
  ParaLens' mmp (t '[batch, i]) (t '[batch, o])
matMulLens = repara r matMulLensCore
  where
    r :: Lens' mmp mmp
    r = alongside gradUpdate gradUpdate
{-# INLINE matMulLens #-}

-- withMomentum :: (Sample f, Num (f R)) => Momentum R -> ParaLens' (MMP, MMP) (f R) (f R)
-- withMomentum mom = repara r matMulLensCore
--   where
--     q :: Lens' ((RM, RM), (RV, RV)) MMP
--     q = alongside mom mom

--     r :: Lens' (MMP, MMP) MMP
--     r = rotate' . q

-- rotate' :: Iso ((a,b),(c,d)) ((a',c'),(b',d')) ((a,c),(b,d)) ((a',b'),(c',d'))
-- rotate' = iso fwd rev
--   where
--     fwd ((a,b),(c,d)) = ((a,c),(b,d))
--     rev ((a,c),(b,d)) = ((a,b),(c,d))

-- matMulLensMom :: (Sample f, Num (f R)) => R -> ParaLens' (MMP, MMP) (f R) (f R)
-- matMulLensMom gamma = withMomentum $ momentum gamma

-- matMulLensNest :: (Sample f, Num (f R)) => R -> ParaLens' (MMP, MMP) (f R) (f R)
-- matMulLensNest gamma = withMomentum $ nesterov gamma

-- matMulLensAda :: (Sample f, Num (f R)) => R -> ParaLens' (MMP, MMP) (f R) (f R)
-- matMulLensAda gamma = withMomentum $ adaGrad gamma

-- matMulLensAdam :: (Sample f, Num (f R)) => R -> R -> R -> ParaLens' (MMP, MMP, MMP) (f R) (f R)
-- matMulLensAdam b1 b2 eps = repara r matMulLensCore
--   where
--     ad :: forall t . (Num (t R), Linear R t, Container t R, Floating (t R)) => Lens' ((t R, t R), t R) (t R)
--     ad = adam b1 b2 eps

--     q :: Lens' (((RM, RM), RM), ((RV, RV), RV)) MMP
--     q = alongside ad ad -- kinda neat that we still maintain the flexibility to spec to matrix or vector here!

--     r :: Lens' (MMP, MMP, MMP) MMP
--     r = rot . q

--     -- lazy type def, actually more general but won't use it elsewhere anyway
--     rot :: Iso' ((a,d),(b,e),(c,f)) (((a,b),c),((d,e),f))
--     rot = iso fwd rev
--       where
--         fwd ((a,d),(b,e),(c,f)) = (((a,b),c),((d,e),f))
--         rev (((a,b),c),((d,e),f)) = ((a,d),(b,e),(c,f))

-- -------------------------
-- -- CONVOLUTION --
-- -------------------------

-- rot180 :: Matrix Double -> Matrix Double
-- rot180 = fliprl . flipud

-- convolve2D :: ParaLens' (Matrix Double) (Matrix Double) (Matrix Double)
-- convolve2D = lens (uncurry corr2) rev
--   where
--     -- fwd :: (Matrix Double, Matrix Double) -> Matrix Double
--     -- fwd (k, a) = corr2 k a

--     rev :: (Matrix Double, Matrix Double) -> Matrix Double -> (Matrix Double, Matrix Double)
--     rev (k, a) dy = (dk, da)
--       where
--         dk = corr2 dy a
--         da = conv2 (rot180 k) dy

-- type Image = [Matrix Double]
-- type Kernels = [[Matrix Double]]        -- [out_channel][in_channel]

-- correlate2D :: ParaLens' Kernels Image Image
-- correlate2D = lens fwd rev
--   where
--     fwd (ks, img) =
--         [ foldl1 add [ corr2 k i | (k, i) <- zip ks_o img ]
--         | ks_o <- ks ]

--     rev (ks, img) dy = (dks, dimg)
--       where
--         dks  = [ [ corr2 d i | i <- img ] | d <- dy ]
--         dimg = [ foldl1 add [ conv2 (rot180 (ks_o !! ic)) d
--                              | (ks_o, d) <- zip ks dy ]
--                | ic <- [0..length img - 1] ]

-- maxPool2DChannel :: Int -> Int -> Lens' (Matrix Double) (Matrix Double)
-- maxPool2DChannel kh kw = lens fwd rev
--   where
--     blockIndices m = [(i, j) | i <- [0..rows m `div` kh - 1]
--                               , j <- [0..cols m `div` kw - 1]]

--     getBlock m i j = subMatrix (i*kh, j*kw) (kh, kw) m

--     fwd :: Matrix Double -> Matrix Double
--     fwd m = let oh = rows m `div` kh
--                 ow = cols m `div` kw
--             in (oh><ow) [maxElement (getBlock m i j) | (i,j) <- blockIndices m]

--     rev :: Matrix Double -> Matrix Double -> Matrix Double
--     rev x dy =
--         let updates = do
--                 (i, j) <- blockIndices x
--                 let block    = getBlock x i j
--                     (_pi, pj) = maxIndex block -- shadows pi
--                     grad     = dy `atIndex` (i, j)
--                 return ((i*kh + _pi, j*kw + pj), grad)
--         in accum (konst 0 (rows x, cols x)) (+) updates

-- -- TODO: Smth possible with traverse here to lift the lens up?
-- -- maxPool2D kh kw = traverse . maxPool2DChannel kh kw
-- maxPool2D :: Int -> Int -> Lens' Image Image
-- maxPool2D kh kw = lens fwd rev
--   where
--     cl :: Lens' (Matrix Double) (Matrix Double)
--     cl = maxPool2DChannel kh kw
--     fwd = map (view cl)
--     rev xs dys = zipWith (set cl) dys xs -- TODO: is the order right?