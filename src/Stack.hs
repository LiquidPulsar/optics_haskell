{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE QuantifiedConstraints #-}

module Stack where
import Core
import Data.Kind
import GHC.TypeLits
import qualified Torch.Typed as T


-- Internal Peano machinery (unchanged)
data PeanoNat = One | Succ PeanoNat

type family Stacked (n :: PeanoNat) (m :: Type) where
    Stacked One      m = m
    Stacked (Succ n) m = (m, Stacked n m)

class StackN (n :: PeanoNat) where
    stack :: ParaLens' p a a -> ParaLens' (Stacked n p) a a

instance StackN 'One where
    stack l = l

instance StackN n => StackN (Succ n) where
    stack :: StackN n => ParaLens' p a a -> ParaLens' (Stacked (Succ n) p) a a
    stack l = l .#. stack @n l

-- Conversion bridge
type family ToPeano (n :: Nat) :: PeanoNat where
    ToPeano 1 = One
    ToPeano n = Succ (ToPeano (n - 1))

-- Public API using numeric literals
type StackedN (n :: Nat) m = Stacked (ToPeano n) m

type CanStack n = StackN (ToPeano n)

stackN :: forall n p a. CanStack n => ParaLens' p a a -> ParaLens' (StackedN n p) a a
stackN = stack @(ToPeano n)

-- x = stackN @5

class RandInit a where
    randInit :: IO a

instance (T.KnownDType dt, T.RandDTypeIsValid dv dt, T.KnownDevice dv, T.TensorOptions s dt dv) => RandInit (T.Tensor dv dt s) where
    randInit = T.randn

instance (RandInit l, RandInit r) => RandInit (l,r) where
    randInit = liftA2 (,) randInit randInit

type RandStack n a = RandInit (Stacked (ToPeano n) a)