{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE AllowAmbiguousTypes #-}

module Static.Attention where

import qualified Torch.Typed as T
import Torch.Typed (Tensor, natValI)
import GHC.TypeLits
import Core
import Control.Lens

type SelfAttnP dev dt e =
  ( Tensor dev dt [e, e]  -- Wq
  , Tensor dev dt [e, e]  -- Wk
  , Tensor dev dt [e, e]  -- Wv
  , Tensor dev dt [e, e]  -- Wo
  )

selfAttention ::
  forall b s e dev dt t.
  ( t ~ Tensor dev dt
  , T.All KnownNat [b, s, e]
  , T.MatMulDTypeIsValid dev dt
  , T.BasicArithmeticDTypeIsValid dev dt
  , T.StandardFloatingPointDTypeValidation dev dt
  , T.KnownDType dt
  , T.KnownDevice dev
  , T.SumDType dt ~ dt
  -- Implied but ah well
  , (b * (s * e)) ~ ((b * s) * e)
  , KnownNat (b * s), T.SumDTypeIsValid dev dt
  ) =>
  ParaLens'
    (SelfAttnP dev dt e)
    (t [b, s, e])
    (t [b, s, e])
selfAttention = lens fwd rev
  where
    scale  = 1.0 / sqrt (fromIntegral (natValI @e)) :: Double
    scaleF = realToFrac scale :: Float
    -- x @ W^T
    proj w x = T.matmul x (T.transpose @0 @1 w)

    -- recomputed in rev, same pattern as sigmoid/relu
    runFwd (wq, wk, wv, wo) x =
      let q       = proj wq x
          k       = proj wk x
          v       = proj wv x
          scores  = T.mulScalar scale $ T.matmul q $ tr k
          weights = T.softmax @2 scores   -- [b, s, s]
          attn    = T.matmul weights v    -- [b, s, e]
      in (q, k, v, weights, attn, proj wo attn)

    fwd (p :: SelfAttnP dev dt e, x) = let (_, _, _, _, _, out) = runFwd p x in out

    rev (p@(wq,wk,wv,wo) :: SelfAttnP dev dt e, x) dOut = ((dWq, dWk, dWv, dWo), dX)
      where
        (q, k, v, weights, attn, _) = runFwd p x

        -- ∂L/∂attn and ∂L/∂Wo via output projection (out = attn @ Wo^T)
        dAttn = T.matmul dOut wo                                            -- [b,s,e] @ [e,e]
        dWo   = wGrad attn dOut

        -- ∂L/∂weights and ∂L/∂v via weighted sum (attn = weights @ v)
        dWeights = mm dAttn      (tr v)                              -- [b,s,s]
        dV       = mm (tr weights) dAttn                             -- [b,s,e]

        -- ∂L/∂scores via softmax
        dScores  = softmaxBwd weights dWeights

        -- ∂L/∂q and ∂L/∂k via scaled matmul (scores = scale * q @ k^T)
        dQ = scale' $ mm dScores      k                             -- [b,s,e]
        dK = scale' $ mm (tr dScores) q                             -- [b,s,e]

        -- ∂L/∂Wq,Wk,Wv and ∂L/∂x via input projections (q = x @ Wq^T)
        dWq = wGrad x dQ
        dWk = wGrad x dK
        dWv = wGrad x dV
        dX  = T.matmul dQ wq + T.matmul dK wk + T.matmul dV wv

    -- Batched matmul shorthand
    mm :: t [b, s1, n] -> t [b, n, s2] -> t [b, s1, s2]
    mm = T.matmul

    -- Transpose last two dims
    tr :: t [b, x, y] -> t [b, y, x]
    tr = T.transpose @1 @2

    -- Scale by 1/sqrt(e)
    scale' :: t [b, s, e] -> t [b, s, e]
    scale' = T.mulScalar scaleF

    -- Weight gradient: for Y = X @ W^T, dW = dY^T @ X (flattened over batch*seq)
    wGrad :: t [b, s, e] -> t [b, s, e] -> t [e, e]
    wGrad x dY = T.matmul (T.transpose @0 @1 (T.reshape @[b * s, e] dY)) (T.reshape @[b * s, e] x)

    -- Softmax backward: dX_i = Y_i * (dY_i - Σ_j Y_j dY_j)
    softmaxBwd :: t [b, s, s] -> t [b, s, s] -> t [b, s, s]
    softmaxBwd w dw = w * T.sub dw dot
      where
        dot = T.reshape @[b, s, 1] $ T.sumDim @2 (w * dw)

multiHeadSelfAttention ::
  forall b s e h hd dev dt t.
  ( t ~ Tensor dev dt
  , T.All KnownNat '[b, s, e, h, hd]
  , e ~ h * hd
  , T.MatMulDTypeIsValid dev dt
  , T.BasicArithmeticDTypeIsValid dev dt
  , T.StandardFloatingPointDTypeValidation dev dt
  , T.KnownDType dt, T.KnownDevice dev
  , T.SumDType dt ~ dt, T.SumDTypeIsValid dev dt
  , KnownNat (b * s)
  , (b * (s * e)) ~ ((b * s) * e)              -- wGrad reshape
  , T.Numel '[b, s, e] ~ T.Numel '[b, s, h, hd] -- splitHeads reshape
  ) =>
  ParaLens' (SelfAttnP dev dt e) (t '[b, s, e]) (t '[b, s, e])
multiHeadSelfAttention = lens fwd rev
  where
    scale  = 1.0 / sqrt (fromIntegral (natValI @hd)) :: Double  -- hd not e
    scaleF = realToFrac scale :: Float

    proj w x = T.matmul x (T.transpose @0 @1 w)

    -- [b, s, e] <-> [b, h, s, hd]
    splitHeads :: t '[b, s, e]    -> t '[b, h, s, hd]
    splitHeads  = T.transpose @1 @2 . T.reshape @'[b, s, h, hd]
    mergeHeads :: t '[b, h, s, hd] -> t '[b, s, e]
    mergeHeads  = T.reshape @'[b, s, e] . T.transpose @1 @2

    -- 4D helpers (extra h dim)
    mmH :: t '[b, h, s1, n] -> t '[b, h, n, s2] -> t '[b, h, s1, s2]
    mmH = T.matmul
    trH :: t '[b, h, x, y] -> t '[b, h, y, x]
    trH = T.transpose @2 @3
    scaleH :: t '[b, h, s, hd] -> t '[b, h, s, hd]
    scaleH = T.mulScalar scaleF

    softmaxBwdH :: t '[b, h, s, s] -> t '[b, h, s, s] -> t '[b, h, s, s]
    softmaxBwdH w dw = w * T.sub dw dot
      where dot = T.reshape @'[b, h, s, 1] $ T.sumDim @3 (w * dw)  -- dim 3, not 2

    wGrad :: t '[b, s, e] -> t '[b, s, e] -> t '[e, e]
    wGrad x dY = T.matmul (T.transpose @0 @1 (T.reshape @'[b*s, e] dY))
                          (T.reshape @'[b*s, e] x)

    runFwd (wq, wk, wv, wo) x =
      let q       = splitHeads $ proj wq x         -- [b, h, s, hd]
          k       = splitHeads $ proj wk x
          v       = splitHeads $ proj wv x
          scores  = T.mulScalar scale $ mmH q (trH k)  -- [b, h, s, s]
          weights = T.softmax @3 scores                 -- dim 3, not 2
          attn    = mergeHeads $ mmH weights v          -- [b, s, e]
      in (q, k, v, weights, attn, proj wo attn)

    fwd (p, x) = let (_, _, _, _, _, out) = runFwd p x in out

    rev (p@(wq,wk,wv,wo), x) dOut = ((dWq, dWk, dWv, dWo), dX)
      where
        (q, k, v, weights, attn, _) = runFwd p x

        -- output projection backward (still 3D)
        dAttn3D = T.matmul dOut wo
        dWo     = wGrad attn dOut

        -- split gradient into heads for attention backward
        dAttn   = splitHeads dAttn3D              -- [b, h, s, hd]

        -- weighted sum backward (4D now)
        dWeights = mmH dAttn   (trH v)            -- [b, h, s, s]
        dV       = mmH (trH weights) dAttn        -- [b, h, s, hd]

        -- softmax backward (dim 3)
        dScores  = softmaxBwdH weights dWeights

        -- scaled matmul backward
        dQ = scaleH $ mmH dScores      k          -- [b, h, s, hd]
        dK = scaleH $ mmH (trH dScores) q

        -- merge heads before weight gradients and dX
        dQ3D = mergeHeads dQ
        dK3D = mergeHeads dK
        dV3D = mergeHeads dV

        dWq = wGrad x dQ3D
        dWk = wGrad x dK3D
        dWv = wGrad x dV3D
        dX  = T.matmul dQ3D wq + T.matmul dK3D wk + T.matmul dV3D wv