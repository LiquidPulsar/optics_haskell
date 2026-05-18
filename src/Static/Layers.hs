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
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE BangPatterns #-}
{-# OPTIONS_GHC -Wno-unused-local-binds #-}

module Static.Layers where

import Control.Arrow
import Control.Lens
import Control.Monad
import Core
import Data.Proxy
import GHC.TypeNats
import Static.Optim
import qualified Torch as U
import qualified Torch.Functional.Internal as I
import Torch.Typed (ConvSideCheck, Fst, Init, Snd, Tensor (UnsafeMkTensor), toDynamic, type (++), natValI)
import qualified Torch.Typed as T
import Types hiding (MMP)
import Static.Bugfix (convTranspose2d, im2col)

-------------------------
-- CARTESIAN REVERSE DIFFERENTIAL CATEGORIES --
-------------------------

-------------------------
-- LAYERS --
-------------------------

transp :: Tensor dv dt shape -> Tensor dv dt (T.SetValue (T.SetValue shape 0 (T.GetValue shape 1)) 1 (T.GetValue shape 0))
transp = T.transpose @0 @1

linear ::
  forall batch i o dv dt.
  ( T.MatMulDTypeIsValid dv dt
  ) =>
  ParaLens'
    (Tensor dv dt '[o, i])
    (Tensor dv dt '[batch, i])
    (Tensor dv dt '[batch, o])
linear = lens fwd rev
  where
    fwd (w, x) = T.matmul x $ transp w

    rev (w, x) grad = (dW, dx)
      where
        dW = T.matmul (transp grad) x
        dx = T.matmul grad w
{-# INLINE linear #-}

linear' ::
  forall t i o shape dv dt.
  ( t ~ Tensor dv dt,
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

type CanAddLens dv dt =
  ( T.BasicArithmeticDTypeIsValid dv dt,
    T.SumDTypeIsValid dv dt,
    T.SumDType dt ~ dt
  )

addLens ::
  forall shape b dv dt t.
  ( t ~ Tensor dv dt,
    CanAddLens dv dt,
    b : shape ~ T.Broadcast shape (b : shape) -- trivial tbh
  ) =>
  ParaLens' (t shape) (t (b : shape)) (t (b : shape))
addLens = lens fwd rev
  where
    fwd :: (t shape, t (b : shape)) -> t (b : shape)
    fwd = uncurry T.add
    rev :: (t shape, t (b : shape)) -> t (b : shape) -> (t shape, t (b : shape))
    rev _ = T.sumDim @0 &&& id
{-# INLINE addLens #-}

-- Note: Can be better with fully known shapes
addLens' ::
  forall shape shape' shape'' dv dt t.
  ( t ~ Tensor dv dt,
    CanAddLens dv dt,
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
{-# INLINEABLE addLens' #-}

sigmoid ::
  forall shape dv dt t.
  ( t ~ Tensor dv dt,
    T.StandardFloatingPointDTypeValidation dv dt,
    T.KnownDevice dv
  ) =>
  Lens' (t shape) (t shape)
sigmoid = lens T.sigmoid rev
  where
    rev :: t shape -> t shape -> t shape
    rev = (*) . ap (*) (1 -) . T.sigmoid
{-# INLINE sigmoid #-}

relu ::
  forall shape dv dt t.
  ( t ~ Tensor dv dt,
    shape ~ T.Broadcast shape shape,
    T.StandardFloatingPointDTypeValidation dv dt,
    T.KnownDevice dv,
    T.ComparisonDTypeIsValid dv dt,
    T.KnownDType dt
  ) =>
  Lens' (t shape) (t shape)
relu = lens T.relu rev
  where
    rev :: t shape -> t shape -> t shape
    rev = (*) . heaviside
{-# INLINE relu #-}

heaviside ::
  forall shape dv dt t.
  ( t ~ Tensor dv dt,
    shape ~ T.Broadcast shape shape,
    T.StandardFloatingPointDTypeValidation dv dt,
    T.ComparisonDTypeIsValid dv dt,
    T.KnownDType dt
  ) =>
  t shape ->
  t shape
-- TODO: pretty sure there is better using geScalar...
heaviside = T.toDType @dt @T.Bool . liftA2 ($) T.gt T.zerosLike

-------------------------
-- MATMUL & OPTIMISED  --
-------------------------

type CanMMLens dv dt = (T.MatMulDTypeIsValid dv dt, CanAddLens dv dt)

matMulLensCore ::
  forall t batch i o dv dt.
  ( t ~ Tensor dv dt,
    CanMMLens dv dt
  ) =>
  ParaLens' (t '[o, i], t '[o]) (t '[batch, i]) (t '[batch, o])
matMulLensCore = linear .#. addLens
{-# INLINE matMulLensCore #-}

type MMP dv dt o i = (Tensor dv dt '[o, i], Tensor dv dt '[o])

matMulLens ::
  forall t mmp batch i o dv dt.
  ( t ~ Tensor dv dt,
    mmp ~ MMP dv dt o i,
    CanMMLens dv dt,
    T.KnownDevice dv
  ) =>
  ParaLens' mmp (t '[batch, i]) (t '[batch, o])
matMulLens = repara r matMulLensCore
  where
    r :: Lens' mmp mmp
    r = alongside gradUpdate gradUpdate
{-# INLINE matMulLens #-}

withMomentum ::
  forall t batch i o dv dt.
  ( t ~ Tensor dv dt,
    CanMMLens dv dt,
    T.KnownDevice dv,
    T.StandardFloatingPointDTypeValidation dv dt
  ) =>
  Momentum ->
  ParaLens' (Two (MMP dv dt o i)) (t '[batch, i]) (t '[batch, o])
withMomentum mom = repara r matMulLensCore
  where
    q :: Lens' (Two (Tensor dv dt '[o, i]), Two (Tensor dv dt '[o])) (MMP dv dt o i)
    q = alongside mom mom

    r :: Lens' (Two (MMP dv dt o i)) (MMP dv dt o i)
    r = rotate' . q
{-# INLINE withMomentum #-}

rotate' :: Iso ((a, b), (c, d)) ((a', c'), (b', d')) ((a, c), (b, d)) ((a', b'), (c', d'))
rotate' = iso fwd rev
  where
    fwd ((a, b), (c, d)) = ((a, c), (b, d))
    rev ((a, c), (b, d)) = ((a, b), (c, d))

matMulLensMom,
  matMulLensNest ::
    ( T.Scalar a,
      Num a,
      CanMMLens dv dt,
      T.StandardFloatingPointDTypeValidation dv dt,
      T.KnownDevice dv
    ) =>
    a ->
    ParaLens' (Two (MMP dv dt o i)) (T.Tensor dv dt '[b, i]) (T.Tensor dv dt '[b, o])
matMulLensMom gamma = withMomentum $ momentum gamma
{-# INLINE matMulLensMom #-}
matMulLensNest gamma = withMomentum $ nesterov gamma
{-# INLINE matMulLensNest #-}

matMulLensAda ::
  ( T.Scalar a,
    Num a,
    Fractional a,
    CanMMLens dv dt,
    T.StandardFloatingPointDTypeValidation dv dt,
    T.KnownDevice dv
  ) =>
  a ->
  ParaLens' (Two (MMP dv dt o i)) (T.Tensor dv dt '[b, i]) (T.Tensor dv dt '[b, o])
matMulLensAda gamma = withMomentum $ adaGrad gamma
{-# INLINE matMulLensAda #-}

matMulLensAdam ::
  forall a dv dt o i b.
  ( T.Scalar a,
    Num a,
    Fractional a,
    CanMMLens dv dt,
    T.StandardFloatingPointDTypeValidation dv dt,
    T.KnownDevice dv
  ) =>
  a ->
  a ->
  a ->
  ParaLens' (Three (MMP dv dt o i)) (T.Tensor dv dt '[b, i]) (T.Tensor dv dt '[b, o])
matMulLensAdam b1 b2 eps = repara r matMulLensCore
  where
    ad :: Momentum2
    ad = adam b1 b2 eps

    -- q :: Lens' (((RM, RM), RM), ((RV, RV), RV)) MMP
    q :: Lens' (TwoNOne (Tensor dv dt '[o, i]), TwoNOne (Tensor dv dt '[o])) (MMP dv dt o i)
    q = alongside ad ad -- kinda neat that we still maintain the flexibility to spec to matrix or vector here!
    r :: Lens' (Three (MMP dv dt o i)) (MMP dv dt o i)
    r = rot . q
{-# INLINE matMulLensAdam #-}

-- lazy type def, actually more general but won't use it elsewhere anyway
rot :: Iso' ((a, d), (b, e), (c, f)) (((a, b), c), ((d, e), f))
rot = iso fwd rev
  where
    fwd ((a, d), (b, e), (c, f)) = (((a, b), c), ((d, e), f))
    rev (((a, b), c), ((d, e), f)) = ((a, d), (b, e), (c, f))
{-# INLINE rot #-}

-- No padding, stride 1 — mirrors the HMatrix corr2/conv2 pair
convLens ::
  forall inC outC batch h w dv dt t oH oW kH kW.
  ( t ~ Tensor dv dt,
    T.All KnownNat '[kH, kW, outC],
    ConvSideCheck h kH 1 0 oH,
    ConvSideCheck w kW 1 0 oW,
    T.All KnownNat '[inC, outC, h, w, batch, oH, oW],
    -- req'd by backwards pass dx calc
    -- h ~ ((oH - 1) + kH),
    -- w ~ ((oW - 1) + kW),
    -- 1 <= h,
    -- 1 <= w,
    -- kH -1 <= oH,
    -- kW -1 <= oW,
    -- bottom 2 only req'd by the T.zeros
    T.KnownDType dt,
    T.KnownDevice dv
  ) =>
  ParaLens'
    (t '[outC, inC, kH, kW]) -- kernel
    (t '[batch, inC, h, w]) -- input
    (t '[batch, outC, oH, oW]) -- output
convLens = lens fwd rev
  where
    fwd :: (t '[outC, inC, kH, kW], t '[batch, inC, h, w]) -> t '[batch, outC, oH, oW]
    fwd (kernel, x) = T.conv2d @'(1, 1) @'(0, 0) kernel T.zeros x

    rev :: (t '[outC, inC, kH, kW], t '[batch, inC, h, w]) -> t '[batch, outC, oH, oW] ->(t '[outC, inC, kH, kW], t '[batch, inC, h, w])
    rev (kernel, x) grad = (dW, dx)
      where
        inC'  = natValI @inC
        outC' = natValI @outC
        kh    = natValI @kH
        kw    = natValI @kW
        oh    = natValI @oH
        ow    = natValI @oW

        opts = U.withDType (T.dtypeVal @dt) . U.withDevice (T.deviceVal @dv) $ U.defaultOpts

        -- dx: transposed convolution — single tensor, no tuple
        dx :: t '[batch, inC, h, w]
        dx = convTranspose2d @'(1,1) @'(0,0) kernel T.zeros grad

        -- dW: im2col unfolds x into patches, then bmm contracts over spatial dims
        -- avoiding convolution_backward_overrideable entirely
        dW :: t '[outC, inC, kH, kW]
        dW = 
          let xU       = toDynamic x
              gradU    = toDynamic grad
              -- [batch, inC*kH*kW, oH*oW]
              xCol     = im2col xU (kh,kw) (1,1) (0,0) (1,1)
              -- [batch, outC, oH*oW]
              -- [b,_,oh,ow] = U.shape gradU
              gradFlat = U.reshape [-1, outC', oh*ow] gradU
              -- [batch, outC, inC*kH*kW]
              dWAll    = I.bmm gradFlat (U.transpose (U.Dim 1) (U.Dim 2) xCol)
          -- in T.zeros
          in UnsafeMkTensor $ U.reshape [outC', inC', kh, kw] $
              I.sumDim dWAll 0 False (U.dtype xU)
{-# INLINE convLens #-}

maxPool ::
  forall kernelSize stride padding channels h w batch oH oW dtype device t.
  ( t ~ Tensor device dtype,
    T.All KnownNat '[Fst kernelSize, Snd kernelSize, Fst stride, Snd stride, Fst padding, Snd padding, channels, h, w, batch],
    ConvSideCheck h (Fst kernelSize) (Fst stride) (Fst padding) oH,
    ConvSideCheck w (Snd kernelSize) (Snd stride) (Snd padding) oW, 
    T.KnownDType dtype
  ) =>
  Lens' (t '[batch, channels, h, w]) (t '[batch, channels, oH, oW])
maxPool = lens fwd rev
  where
    fwd :: t '[batch, channels, h, w] -> t '[batch, channels, oH, oW]
    fwd = T.maxPool2d @kernelSize @stride @padding

    rev :: t '[batch, channels, h, w] -> t '[batch, channels, oH, oW] -> t '[batch, channels, h, w]
    rev x grad = UnsafeMkTensor $
      let kh = T.natValI @(Fst kernelSize)
          kw = T.natValI @(Snd kernelSize)
          h' = T.natValI @h
          w' = T.natValI @w
          xU    = toDynamic x
          gradU = toDynamic grad
          -- recompute forward to get max values
          n     = toDynamic (fwd x)
          -- repeat max values and gradient back to input spatial size
          expand t = I.repeat_interleave_tlll
                      (I.repeat_interleave_tlll t kh 2 h')
                      kw 3 w'
          mask  = U.toType (T.dtypeVal @dtype) $ U.eq xU (expand n)
      in U.mul mask (expand gradU)
    -- rev x grad = UnsafeMkTensor $
    --   let kh = T.natValI @(Fst kernelSize)
    --       kw = T.natValI @(Snd kernelSize)
    --       sh = T.natValI @(Fst stride)
    --       sw = T.natValI @(Snd stride)
    --       ph = T.natValI @(Fst padding)
    --       pw = T.natValI @(Snd padding)
    --       h' = T.natValI @h
    --       w' = T.natValI @w
    --       (_, inds) = I.max_pool2d_with_indices
    --         (toDynamic x) (kh,kw) (sh,sw) (ph,pw) (1,1) False -- dilation 1,1
    --   in I.max_unpool2d (toDynamic grad) inds (h', w')
{-# INLINE maxPool #-}

-- ─── Flatten lens ─────────────────────────────────────────────────────────────

-- Flattens all dims after batch into one, using ShapeProduct (:: Nat) to avoid
-- the Natural/Nat kind mismatch that T.Numel and T.Product both have.
flatten ::
  forall batch shape dev dt t.
  ( t ~ Tensor dev dt
  , KnownNat batch
  , T.KnownShape shape
  , KnownNat (T.Product shape)
  ) =>
  Lens' (t (batch : shape)) (t '[batch, T.Product shape])
flatten = lens fwd rev
  where
    b    = natValI @batch
    flat = natValI @(T.Product shape)
    dims = T.shapeVal @shape          -- runtime shape for rev reshape
    fwd x   = UnsafeMkTensor $ U.reshape [b, flat] (toDynamic x) -- use -1 here to save the flat?
    rev _ g = UnsafeMkTensor $ U.reshape (b : dims) (toDynamic g)
{-# INLINE flatten #-}