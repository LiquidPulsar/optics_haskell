{-# LANGUAGE RankNTypes #-}
module MnistCNN where

import Numeric.LinearAlgebra
import Data.Time (getCurrentTime, diffUTCTime)
import System.IO (hSetBuffering, stdout, BufferMode(..))
import qualified Data.Vector.Unboxed as VU
import Control.Lens
import Data.IDX

import Core
import Types
import Layers
import Loss
import Optim
import Models
import Data.Foldable
import Iris (initParams)
import Data.Maybe
import Control.Arrow


-- (conv1 kernels, conv2 kernels, dense params)
type MnistParams = (Kernels, (Kernels, MMP))

-- ─── Image-level activation / pooling ────────────────────────────────────────

reluImg :: Lens' Image Image
reluImg = lens (map $ cmap (max 0)) rev
  where rev = zipWith ((*) . step)
    -- rev img dy = zipWith (\i d -> step i * d) img dy

maxPool2DImg :: Int -> Int -> Lens' Image Image
maxPool2DImg kh kw = lens fwd rev
  where
    cl :: Lens' (Matrix Double) (Matrix Double)
    cl = maxPool2DChannel kh kw
    fwd           = map (view cl)
    rev = zipWith (flip $ set cl)
    -- rev xs dys    = zipWith (\x dy -> set cl dy x) xs dys

-- ─── Flatten ─────────────────────────────────────────────────────────────────

-- Lens' so it carries no parameters — use toPara to lift into Para
flattenImg :: Lens' Image RV
flattenImg = lens fwd rev
  where
    fwd = vjoin . map flatten

    rev img dv =
        let shapes = map (rows &&& cols) img
            sizes  = map (uncurry (*)) shapes
            vecs   = takesV sizes dv
        in zipWith (reshape . snd) shapes vecs
        -- in zipWith (\(r, c) v -> reshape c v) shapes vecs

-- ─── Model ───────────────────────────────────────────────────────────────────

-- Input:  1 channel  28x28
-- conv1:  3 channels 26x26  (3x3 kernel)
-- pool1:  3 channels 13x13
-- conv2:  5 channels 10x10  (4x4 kernel)
-- pool2:  5 channels  5x5
-- flatten:           125
-- dense:             10

l :: (Num p) => Lens' [[p]] [[p]]
l = liftUpdate $ liftUpdate gradUpdate

x :: ParaLens' Kernels Image Image
x = repara l correlate2D

mnistModel :: ParaLens' (Inp Image, MnistParams) () (Out RV)
mnistModel = argToPara
    .#. x . reluImg . maxPool2DImg 2 2
    .#. x . reluImg . maxPool2DImg 2 2
    .#. final
    where
      final :: ParaLens' MMP Image RV
      final = rightLens flattenImg . matMulLens . sigmoid

mnistModelLoss :: ParaLens' ((Inp Image, MnistParams), Tgt RV) () R
mnistModelLoss = mnistModel .#. lossSmooth

mnistModel' :: ParaLRLens' ((Inp Image, MnistParams), Tgt RV) ()
mnistModel' = mnistModel .#. lossSmooth . lrSmooth 0.01

-- ─── Initialisation ──────────────────────────────────────────────────────────

initKernels :: Int -> Int -> Int -> Int -> IO Kernels
initKernels kh kw inC outC =
    sequence
        [ sequence
            [scale 0.01 . cmap (subtract 0.5) <$> rand kh kw
            | _ <- [1..inC] ]
        | _ <- [1..outC] ]

mnistInitParams :: IO MnistParams
mnistInitParams = do
    k1  <- initKernels 3 3 1 3
    k2  <- initKernels 4 4 3 5
    dense <- initParams 125 10       -- from Iris module
    return (k1, (k2, dense))

-- ─── MNIST loading ───────────────────────────────────────────────────────────

mnistToImage :: Matrix Double -> Image
mnistToImage m = [m]  -- single channel, 28x28

loadMnist :: FilePath -> FilePath -> IO [(Image, RV)]
loadMnist imgPath lblPath = do
    Just imgs <- decodeIDXFile imgPath
    Just lbls <- decodeIDXLabelsFile lblPath
    let pairs = fromJust $ labeledDoubleData lbls imgs
    return [ (mnistToImage $ reshape 28 $ fromList $ map realToFrac $ VU.toList xs, oneHot 10 l')
           | (l', xs) <- pairs ] -- l' to avoid shadowing
  where
    oneHot n i = fromList [ if j == i then 1 else 0 | j <- [0..n-1] ]

-- ─── Training ────────────────────────────────────────────────────────────────

chunksOf :: Int -> [a] -> [[a]]
chunksOf _ [] = []
chunksOf n xs = take n xs : chunksOf n (drop n xs)

mnistEpoch :: Int -> [(Image, RV)] -> MnistParams -> MnistParams
mnistEpoch batch tgts p = foldl (trainMany mnistModel') p $ chunksOf batch tgts

mnistPredict :: MnistParams -> Image -> Int
mnistPredict p img = maxIndex $ runFullModel mnistModel (img, p)

mnistAccuracy :: MnistParams -> [(Image, RV)] -> Double
mnistAccuracy p targets = correct / total
  where
    correct = fromIntegral . length . filter id $
        [ mnistPredict p img == maxIndex tgt | (img, tgt) <- targets ]
    total   = fromIntegral (length targets)

mnistTrain :: IO ()
mnistTrain = do
    hSetBuffering stdout LineBuffering
    train <- loadMnist "data/train-images-idx3-ubyte" "data/train-labels-idx1-ubyte"
    -- print $ length train
    test  <- loadMnist "data/t10k-images-idx3-ubyte"  "data/t10k-labels-idx1-ubyte"
    p0    <- mnistInitParams
    t0  <- getCurrentTime
    let epochs = iterate (mnistEpoch 32 $ take 64 train) p0
    forM_ (zip [0..] epochs) $ \(e, p) -> do
        let acc = mnistAccuracy p $ take 64 test
        t1  <- getCurrentTime
        let (_, (_, (w, _))) = p  -- dense layer weights as a proxy
        putStrLn $ unwords 
            [ "epoch", show (e :: Int)
            , "\taccuracy", show acc
            , "\ttime", show (diffUTCTime t1 t0)
            , "\tmean_w", show (sumElements w / fromIntegral (rows w * cols w))
            ]