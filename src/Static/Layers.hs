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
-- I can't import a type (*) as qualified??
-- import qualified GHC.TypeLits as TL (type (+))

import Data.Proxy
import GHC.TypeNats
import Optim
import qualified Torch as U
import Torch.Typed (Init, Tensor (UnsafeMkTensor), toDynamic, type (++))
import qualified Torch.Typed as T

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
-- linear :: (Sample f, Floating e, Numeric e) => ParaLens' (Matrix e) (f e) (f e)
-- linear = lens (uncurry applyLinear) rev
--   where rev (m, x) y = (accumGrad y x, tr m `applyLinear` y)

transp :: Tensor device dtype shape -> Tensor device dtype (T.SetValue (T.SetValue shape 0 (T.GetValue shape 1)) 1 (T.GetValue shape 0))
transp = T.transpose @0 @1

linear ::
  forall batch i o device dtype.
  ( T.All KnownNat [i, o, batch],
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

-- type Tensor = Tensor

-- type family Init (xs :: [Nat]) :: [Nat] where
--   Init '[_]      = '[]
--   Init (x ': xs) = x ': Init xs

-- sumSamples'
--   :: forall shape o batch device dtype
--    . ( o ~ T.Last shape
--     --  , KnownNat o
--      , T.KnownShape shape
--      , batch ~ (T.Numel shape `Div` o)
--     --  , KnownNat batch
--      , T.SumDType dtype ~ dtype, T.SumDTypeIsValid device dtype
--      , T.Numel shape ~ T.Product (o ': Init shape)
--      )
--   => Tensor device dtype shape
--   -> Tensor device dtype '[o]
-- -- sumSamples' = sumDim @0 . reshape @('[batch, o] :: [Nat])
-- sumSamples' t =
--   let flat = T.reshape t :: Tensor device dtype '[T.Product (Init shape), o]
--   in  T.sumDim @0 flat

-- A simple linear layer: y = x @ W^T + b
-- myLinear
--   :: forall batchSize inFeatures outFeatures device dtype
--    . ( T.All KnownNat '[batchSize, inFeatures, outFeatures]
--      , T.KnownDevice device
--      , T.MatMulDTypeIsValid device dtype
--      , T.StandardFloatingPointDTypeValidation device dtype )
--   => Tensor device dtype '[outFeatures, inFeatures]  -- weight W
--   -> Tensor device dtype '[outFeatures]               -- bias b
--   -> Tensor device dtype '[batchSize, inFeatures]     -- input x
--   -> Tensor device dtype '[batchSize, outFeatures]    -- output y
-- myLinear weight bias input =
--   -- matmul: [b, i] @ [i, o] = [b, o]
--   -- then broadcast-add bias [o]
--   T.matmul input (T.transpose @0 @1 weight) + bias

-- addLens'
--   :: forall shape o device dtype t
--    . ( t ~ Tensor device dtype -- for neatness
--      , KnownNat o
--     --  , T.IsSuffixOf '[o] shape
--      , T.Last shape ~ o
--      , T.BasicArithmeticDTypeIsValid device dtype
--      , T.SumDTypeIsValid device dtype
--      , T.SumDType dtype ~ dtype
--      , shape ~ T.Broadcast '[o] shape -- This is implied by T.Last shape ~ o
--      )
--   => ParaLens' (t '[o]) (t shape) (t shape)
-- addLens' = lens fwd rev
--   where
--     fwd :: (t '[o], t shape) -> t shape
--     fwd (b,x) = T.add b x
--     rev :: (t '[o], t shape) -> t shape -> (t '[o], t shape)
--     rev _ x' = (foo, x')
--       where
--         foo = sumSamples' x'
--         o' = fromIntegral . natVal $ Proxy @o
--         b = T.numel x' `div` o'

-- addLens''
--   :: forall b o device dtype t
--    . ( t ~ Tensor device dtype -- for neatness
--      , T.All KnownNat '[b, o]
--      , T.BasicArithmeticDTypeIsValid device dtype
--      , T.SumDTypeIsValid device dtype
--      , T.KnownDevice device -- why?
--      , T.SumDType dtype ~ dtype
--      , '[b, o] ~ T.Broadcast '[o] '[b, o] -- This is already implied but ah well
--      )
--   => ParaLens' (t '[o]) (t '[b, o]) (t '[b, o])
-- addLens'' = lens fwd rev
--   where
--     fwd :: (t '[o], t '[b, o]) -> t '[b, o]
--     fwd (b,x) = T.add b x
--     rev :: (t '[o], t '[b, o]) -> t '[b, o] -> (t '[o], t '[b, o])
--     rev _ x' = (T.sumDim @0 x' / b', x')
--       where
--         b' = fromIntegral . natVal $ Proxy @b

type CanAddLens device dtype =
  ( 
    T.BasicArithmeticDTypeIsValid device dtype,
    T.SumDTypeIsValid device dtype,
    T.SumDType dtype ~ dtype
  )

-- Note: Can be better with fully known shapes
addLens ::
  forall shape shape' shape'' device dtype t.
  ( t ~ Tensor device dtype,
    CanAddLens device dtype,
    T.KnownDevice device,
    T.KnownShape shape,
    KnownNat (T.Numel shape), -- implied by KnownShape tbh...
    shape'' ~ T.Broadcast shape shape',
    shape'' ~ shape',
    shape `T.IsSuffixOf` shape'
  ) =>
  ParaLens' (t shape) (t shape') (t shape'')
addLens = lens fwd rev
  where
    fwd :: (t shape, t shape') -> t shape''
    fwd (b, x) = T.add b x
    rev :: (t shape, t shape') -> t shape'' -> (t shape, t shape')
    -- Here, we need to sum dims to compress shape'' into shape
    -- Type family that takes number of dims to chop off?
    -- Also take the product of the first dims
    -- But this may not be a knownNat for batch dim...
    -- Can I use a shape ~ s:ss where T.All KnownNat ss?
    -- i.e. we get one dim unknown to use and the rest
    -- But then actually we might as well say we know only the suffixed part
    -- (which we aready do) and then have to take the runtime size of the extra
    -- Since we have variable batch sizes (as less than 32 etc - but do we care?)
    -- Could mandate known size, or just write a version for known / unknown
    -- If we do that, use OverlappingInstances to specialise:
    -- general shape -> runtime size, knownNat instance -> reify!
    rev _ x' = (x''' / batchSize, x')
      where
        -- Safe as shape is suffix of shape' so multiples will work
        x'' :: t (w : shape) -- There exists a w, doesn't really matter what it is
        x'' = UnsafeMkTensor $ U.reshape (-1 : T.shapeVal @shape) $ toDynamic x'

        x''' :: t shape
        x''' = T.sumDim @0 x''

        -- T.numel b is runtime (for rev (b,_) x'), we can use this for compile-time!
        numelB = fromIntegral $ natVal $ Proxy @(T.Numel shape)
        batchSize = fromIntegral $ T.numel x' `div` numelB -- x' has partial shape runtime-specific

--                            m x y
-- addLens :: forall e f. (Sample f, ProdNum e) => ParaLens' (Vector e) (f e) (f e)
-- addLens = lens (uncurry addBias) rev
--   where
--     rev :: (Vector e, f e) -> f e -> (Vector e, f e)
--     rev = const $ sumSamples &&& id

-- expit :: Floating a => a -> a
-- expit = recip . (1 +) . exp . negate

-- sigmoid :: forall e f. (Sample f, Floating e, Numeric e, Num (f e)) => Lens' (f e) (f e)
-- sigmoid = lens (cmap expit) rev
--   where
--     fwd :: f e -> f e
--     fwd = cmap expit

--     rev :: f e -> f e -> f e
--     rev = (*) . ap (*) (1 -) . fwd

sigmoid ::
  forall shape device dtype t.
  ( t ~ Tensor device dtype,
    shape ~ T.Broadcast shape shape,
    T.StandardFloatingPointDTypeValidation device dtype,
    T.KnownDevice device
  ) =>
  Lens' (t shape) (t shape)
sigmoid = lens T.sigmoid rev
  where
    rev :: t shape -> t shape -> t shape
    rev = (*) . ap (*) (1 -) . T.sigmoid

-- relu :: (Sample f, Floating e, Numeric e, Ord e, Num (f e)) => Lens' (f e) (f e)
-- relu = lens (cmap $ max 0) ((*) . step)

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

-- matMulLensCore :: (Sample f, Num (f R)) => ParaLens' MMP (f R) (f R)
-- matMulLensCore = linear .#. addLens

type CanMMLens device dtype = ( T.MatMulDTypeIsValid device dtype, CanAddLens device dtype)

matMulLensCore ::
  forall t batch i o device dtype.
  ( t ~ Tensor device dtype,
    T.All KnownNat [i, o, batch],
    CanMMLens device dtype,
    T.IsSuffixOf '[o] '[batch, o], -- I'd think this was implied but ah well
    T.KnownDevice device
  ) =>
  ParaLens' (t '[o, i], t '[o]) (t '[batch, i]) (t '[batch, o])
matMulLensCore = linear .#. addLens

type MMP device dtype o i = (Tensor device dtype '[o, i], Tensor device dtype '[o])

matMulLens ::
  forall t mmp batch i o device dtype.
  ( t ~ Tensor device dtype,
    mmp ~ MMP device dtype o i,
    T.All KnownNat [i, o, batch],
    CanMMLens device dtype,
    T.IsSuffixOf '[o] '[batch, o], -- I'd think this was implied but ah well
    T.KnownDevice device
  ) =>
  ParaLens' mmp (t '[batch, i]) (t '[batch, o])
matMulLens = repara r matMulLensCore
  where
    r :: Lens' mmp mmp
    r = alongside gradUpdate gradUpdate

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