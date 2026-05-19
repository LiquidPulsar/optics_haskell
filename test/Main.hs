{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Main (main) where

import Control.Lens (set, view)
import Control.Monad (unless)
import GHC.Generics (Generic)
import qualified Static.Layers as L
import System.Exit (exitFailure)
import Torch
  ( GD (..), Optimizer, Parameter, Parameterized,
    asTensor, asValue, flattenParameters, makeIndependent,
    matmul, reshape, runStep, sumAll, toDependent, transpose2D,
  )
import qualified Torch
import qualified Torch.Typed as T
import Torch.Typed (Tensor (UnsafeMkTensor), toDynamic)

-------------------------
-- HELPERS
-------------------------

type Dev = '(T.CPU, 0)
type Flt = T.Float

-- Build a typed tensor from a flat list of floats.
typed :: [Float] -> [Int] -> Tensor Dev Flt shape
typed vals shape = UnsafeMkTensor $ reshape shape $ asTensor vals

tol :: Float
tol = 1e-4

maxAbsDiff :: Torch.Tensor -> Torch.Tensor -> Float
maxAbsDiff a b = maximum (asValue (Torch.abs (Torch.sub a b)) :: [Float])

check :: String -> Torch.Tensor -> Torch.Tensor -> IO Bool
check name expected actual =
  let d = maxAbsDiff expected actual
  in if d < tol
       then putStrLn ("PASS " <> name) >> return True
       else do
         putStrLn ("FAIL " <> name <> "  maxDiff=" <> show d)
         putStrLn ("  expected: " <> show expected)
         putStrLn ("  actual:   " <> show actual)
         return False

-- With lr=1: new = old - grad  →  grad = old - new
gradViaRunStep
  :: (Parameterized f, Optimizer o)
  => f -> o -> Torch.Tensor -> IO [Torch.Tensor]
gradViaRunStep model optim loss = do
  let oldPs = map toDependent (flattenParameters model)
  (newModel, _) <- runStep model optim loss 1.0
  let newPs = map toDependent (flattenParameters newModel)
  return $ zipWith Torch.sub oldPs newPs

-------------------------
-- testLinear
-- linear :: ParaLens' (T '[o,i]) (T '[b,i]) (T '[b,o])
-- fwd  (w, x) = x @ w^T
-- dW = grad^T @ x
-------------------------

newtype LinearModel = LinearModel { lmW :: Parameter } deriving (Generic, Show)
instance Parameterized LinearModel

testLinear :: IO [Bool]
testLinear = do
  -- b=1 for simplicity; hasktorch reference uses dynamic tensors
  let w0 = reshape [2, 3] $ asTensor ([1,2,3, 4,5,6] :: [Float])
      x0 = reshape [1, 3] $ asTensor ([0.1, 0.2, 0.3] :: [Float])

  wP <- makeIndependent w0
  let mdl = LinearModel wP
      out = matmul x0 (transpose2D (toDependent wP))
      loss = sumAll out
  [dw_ref] <- gradViaRunStep mdl GD loss

  let w     = UnsafeMkTensor w0 :: Tensor Dev Flt '[2, 3]
      x     = UnsafeMkTensor x0 :: Tensor Dev Flt '[1, 3]
      onesG = T.ones @'[1, 2]    :: Tensor Dev Flt '[1, 2]
      y_lens             = view L.linear (w, x)
      (dw_lens, _)       = set  L.linear onesG (w, x)

  sequence
    [ check "linear fwd" (toDynamic y_lens) out
    , check "linear dW"  (toDynamic dw_lens) dw_ref
    ]

-------------------------
-- testAddLens
-- addLens :: ParaLens' (T shape) (T (b:shape)) (T (b:shape))
-- fwd  (bias, x) = x + bias   (broadcast)
-- dBias = sumDim @0 grad
-------------------------

newtype BiasModel = BiasModel { bmBias :: Parameter } deriving (Generic, Show)
instance Parameterized BiasModel

testAddLens :: IO [Bool]
testAddLens = do
  let b0 = asTensor ([10, 20, 30] :: [Float])
      x0 = reshape [4, 3] $ asTensor ([1,2,3, 4,5,6, 7,8,9, 0,1,2] :: [Float])

  bP <- makeIndependent b0
  let mdl  = BiasModel bP
      out  = Torch.add x0 (toDependent bP)  -- x + bias (broadcasts)
      loss = sumAll out
  [db_ref] <- gradViaRunStep mdl GD loss

  let bias  = UnsafeMkTensor b0 :: Tensor Dev Flt '[3]
      x     = UnsafeMkTensor x0 :: Tensor Dev Flt '[4, 3]
      onesG = T.ones @'[4, 3]    :: Tensor Dev Flt '[4, 3]
      y_lens              = view L.addLens (bias, x)
      (dbias_lens, _)     = set  L.addLens onesG (bias, x)

  sequence
    [ check "addLens fwd"   (toDynamic y_lens)    out
    , check "addLens dBias" (toDynamic dbias_lens) db_ref
    ]

-------------------------
-- testSigmoid
-- sigmoid :: Lens' (T shape) (T shape)
-- fwd x = σ(x),  bwd x g = g * σ(x)*(1-σ(x))
-- Treat x as the "parameter" so runStep gives grad w.r.t. x.
-------------------------

newtype InputModel = InputModel { imX :: Parameter } deriving (Generic, Show)
instance Parameterized InputModel

testSigmoid :: IO [Bool]
testSigmoid = do
  let x0 = asTensor ([-2,-1,0,1,2, -0.5,0.5,1.5,-1.5,3] :: [Float])

  xP <- makeIndependent x0
  let mdl  = InputModel xP
      out  = Torch.sigmoid (toDependent xP)
      loss = sumAll out
  [dx_ref] <- gradViaRunStep mdl GD loss

  let x     = UnsafeMkTensor (reshape [2, 5] x0) :: Tensor Dev Flt '[2, 5]
      onesG = T.ones @'[2, 5]                      :: Tensor Dev Flt '[2, 5]
      y_lens  = view L.sigmoid x
      dx_lens = set  L.sigmoid onesG x

  sequence
    [ check "sigmoid fwd" (toDynamic y_lens)  (Torch.sigmoid (reshape [2,5] x0))
    , check "sigmoid bwd" (toDynamic dx_lens) (reshape [2,5] dx_ref)
    ]

-------------------------
-- testRelu
-- relu :: Lens' (T shape) (T shape)
-- fwd x = max(0, x),  bwd x g = g * (x > 0)
-------------------------

testRelu :: IO [Bool]
testRelu = do
  let x0 = asTensor ([-2,-1,0,1,2, -0.5,0.5,1.5,-1.5,3] :: [Float])

  xP <- makeIndependent x0
  let mdl  = InputModel xP
      out  = Torch.relu (toDependent xP)
      loss = sumAll out
  [dx_ref] <- gradViaRunStep mdl GD loss

  let x     = UnsafeMkTensor (reshape [2, 5] x0) :: Tensor Dev Flt '[2, 5]
      onesG = T.ones @'[2, 5]                      :: Tensor Dev Flt '[2, 5]
      y_lens  = view L.relu x
      dx_lens = set  L.relu onesG x

  sequence
    [ check "relu fwd" (toDynamic y_lens)  (Torch.relu (reshape [2,5] x0))
    , check "relu bwd" (toDynamic dx_lens) (reshape [2,5] dx_ref)
    ]

-------------------------
-- testConvForward
-- convLens fwd = T.conv2d @'(1,1) @'(0,0) kernel zeros x
-- Forward only: both sides call identical typed code.
-------------------------

testConvForward :: IO [Bool]
testConvForward = do
  let k = typed (take 18 [1..]) [2,1,3,3] :: Tensor Dev Flt '[2, 1, 3, 3]
      x = typed (take 49 [1..]) [1,1,7,7] :: Tensor Dev Flt '[1, 1, 7, 7]

  let y_lens = view (L.convLens @1 @2 @1 @7 @7) (k, x)
  let y_ref  = T.conv2d @'(1,1) @'(0,0) k T.zeros x

  sequence
    [ check "conv fwd" (toDynamic y_lens) (toDynamic y_ref)
    ]

-------------------------
-- testConvDK
-- Kernel gradient: treat kernel as the parameter; runStep gives dK.
-------------------------

newtype ConvModel = ConvModel { cmKernel :: Parameter } deriving (Generic, Show)
instance Parameterized ConvModel

testConvDK :: IO [Bool]
testConvDK = do
  let k0 = reshape [2,1,3,3] $ asTensor (take 18 [1..] :: [Float])
      x0 = reshape [1,1,7,7] $ asTensor (take 49 [1..] :: [Float])

  kP <- makeIndependent k0
  let mdl  = ConvModel kP
      -- use toDependent so the forward graph is connected to kP
      kT   = UnsafeMkTensor (toDependent kP) :: Tensor Dev Flt '[2, 1, 3, 3]
      xT   = UnsafeMkTensor x0               :: Tensor Dev Flt '[1, 1, 7, 7]
      out  = toDynamic $ T.conv2d @'(1,1) @'(0,0) kT T.zeros xT
      loss = sumAll out
  [dk_ref] <- gradViaRunStep mdl GD loss

  let onesG     = T.ones @'[1, 2, 5, 5] :: Tensor Dev Flt '[1, 2, 5, 5]
      (dk_lens, _) = set (L.convLens @1 @2 @1 @7 @7) onesG (kT, xT)

  sequence
    [ check "conv dK (x25)" (toDynamic dk_lens) dk_ref
    ]

-------------------------
-- testFlatten
-- flatten :: Lens' (T (b:shape)) (T '[b, Product shape])
-- fwd x = reshape,  bwd _ g = reshape back
-------------------------

testFlatten :: IO [Bool]
testFlatten = do
  let x     = typed (take 24 [1..]) [3, 2, 4] :: Tensor Dev Flt '[3, 2, 4]
      onesG = T.ones @'[3, 8]                  :: Tensor Dev Flt '[3, 8]

  let y_lens  = view (L.flatten @3 @'[2, 4]) x
  let dx_lens = set  (L.flatten @3 @'[2, 4]) onesG x

  sequence
    [ check "flatten fwd" (toDynamic y_lens)  (reshape [3, 8]    (toDynamic x))
    , check "flatten bwd" (toDynamic dx_lens) (reshape [3, 2, 4] (toDynamic onesG))
    ]

-------------------------
-- MAIN
-------------------------

main :: IO ()
main = do
  results <- concat <$> sequence
    [ testLinear
    , testAddLens
    , testSigmoid
    , testRelu
    , testConvForward
    , testConvDK
    , testFlatten
    ]
  let passed = length (filter id results)
      total  = length results
  putStrLn $ show passed <> "/" <> show total <> " tests passed"
  unless (passed == total) exitFailure
