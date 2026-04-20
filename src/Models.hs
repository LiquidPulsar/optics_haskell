{-# LANGUAGE TupleSections #-}
{-# LANGUAGE RankNTypes #-}

module Models where
import Core
import Optim
import Control.Lens
import Control.Arrow

runModel :: ParaLens' a b c -> (a, b) -> c
runModel = view

runFullModel :: ParaLens' a () b -> a -> b
runFullModel pl = view pl . (,())

runModelUpdate :: ParaLRLens' a () -> a -> a
runModelUpdate pl = fst . set pl () . (,())

runOne :: ParaLens' ((inp, p), tgt) () p' -> p -> (inp, tgt) -> p'
runOne m p = runFullModel m . first (,p)

trainOne :: ParaLRLens' ((inp, p), tgt) () -> p -> (inp, tgt) -> p
trainOne m p = snd . fst . runModelUpdate m . first (,p)

trainMany :: ParaLRLens' ((inp, p), tgt) () -> p -> [(inp, tgt)] -> p
trainMany m = foldl $ trainOne m