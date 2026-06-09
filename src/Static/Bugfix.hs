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

module Static.Bugfix where

import GHC.TypeNats
import Torch.Typed (ConvSideCheck, Fst, Snd, Tensor, natValI)
import qualified Torch.Typed as T

import GHC.IO (unsafePerformIO)
import Torch.Internal.Cast (cast7, cast5)
import Torch.Internal.Managed.Native (conv_transpose2d_tttllll, im2col_tllll)
import qualified Torch as U


-- https://github.com/hasktorch/hasktorch/blob/186a68c01af7489d1fb23ba66975f5c3570214b5/hasktorch/test/FunctionalSpec.hs#L206

test :: Tensor '(T.CPU, 0) T.Double [10, 10, 5, 6]
test = T.convTranspose2d @'(1,1) @'(0,0) (T.ones @'[3, 10, 1, 1]) (T.ones @'[10]) (T.ones @'[10, 3, 5, 6])

-- test1 :: Tensor '(T.CPU, 0) T.Float [4, 10, 9, 10] -- this doesn't typecheck due to bug :p
-- test1 =
--   T.convTranspose2d @'(1,1) @'(0,0)
--     (T.ones @'[3, 10, 3, 3])   -- weight
--     (T.ones @'[10])            -- bias
--     (T.ones @'[4, 3, 7, 8])    -- input

test1 :: Tensor '(T.CPU, 0) T.Float [4, 10, 9, 10] -- now it does!
test1 =
  convTranspose2d @'(1,1) @'(0,0)
    (T.ones @'[3, 10, 3, 3])   -- weight
    (T.ones @'[10])            -- bias
    (T.ones @'[4, 3, 7, 8])    -- input

convTranspose2d ::
  forall
    (stride :: (Nat, Nat))
    (padding :: (Nat, Nat))
    inputChannelSize
    outputChannelSize
    kernelSize0
    kernelSize1
    inputSize0
    inputSize1
    batchSize
    outputSize0
    outputSize1
    dtype
    device.
  ( T.All
      KnownNat
      [ Fst stride,
         Snd stride,
         Fst padding,
         Snd padding,
         inputChannelSize,
         outputChannelSize,
         kernelSize0,
         kernelSize1,
         inputSize0,
         inputSize1,
         batchSize,
         outputSize0,
         outputSize1
       ],
    -- N.B: convTranspose2D has incorrect constraints in hasktorch, the input and output sizes are flipped
    ConvSideCheck outputSize0 kernelSize0 (Fst stride) (Fst padding) inputSize0,
    ConvSideCheck outputSize1 kernelSize1 (Snd stride) (Snd padding) inputSize1
  ) =>
  -- | weight
  Tensor device dtype [inputChannelSize, outputChannelSize, kernelSize0, kernelSize1] ->
  -- | bias
  Tensor device dtype '[outputChannelSize] ->
  -- | input
  Tensor device dtype [batchSize, inputChannelSize, inputSize0, inputSize1] ->
  -- | output
  Tensor device dtype [batchSize, outputChannelSize, outputSize0, outputSize1]
convTranspose2d weight bias input =
  unsafePerformIO $
    cast7
      conv_transpose2d_tttllll
      input
      weight
      bias
      ([natValI @(Fst stride), natValI @(Snd stride)] :: [Int])
      ([natValI @(Fst padding), natValI @(Snd padding)] :: [Int])
      ([0, 0] :: [Int])
      (1 :: Int)


-- the underlying function takes lists not tuples!
im2col ::
  U.Tensor ->   -- self
  (Int, Int) -> -- kernel_size
  (Int, Int) -> -- dilation
  (Int, Int) -> -- padding
  (Int, Int) -> -- stride
  U.Tensor
im2col self (kh,kw) (dh,dw) (ph,pw) (sh,sw) = unsafePerformIO $
  (cast5 im2col_tllll) self [kh,kw] [dh,dw] [ph,pw] [sh,sw]

test2 :: U.Tensor
test2 = im2col (U.zeros [1,3,5,7] U.defaultOpts) (3, 3) (1, 1) (1, 1) (1, 1)