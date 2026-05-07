{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}

module IrisData (Iris, IrisClass (..), iris, irisClass, petalLength, petalWidth, sepalLength, sepalWidth) where

import System.IO.Unsafe
import Data.Csv
import Data.Char
import qualified Data.ByteString.Lazy as BL
import qualified Data.ByteString.Char8 as BC
import qualified Data.Vector as V
import GHC.Generics
import Control.DeepSeq

data IrisClass = Setosa | Versicolor | Virginica
  deriving (Show, Eq, Ord, Enum, Bounded, Generic, NFData)

data Iris = Iris
  { sepalLength :: Double
  , sepalWidth  :: Double
  , petalLength :: Double
  , petalWidth  :: Double
  , irisClass   :: IrisClass
  } deriving (Generic, Show, FromRecord)

instance FromField IrisClass where
  parseField s = case BC.map toLower s of
    "setosa"     -> pure Setosa
    "versicolor" -> pure Versicolor
    "virginica"  -> pure Virginica
    _            -> fail $ "Unknown iris class: " ++ BC.unpack s

loadIris :: FilePath -> IO (Either String (V.Vector Iris))
loadIris path = decode HasHeader <$> BL.readFile path

iris :: [Iris] -- woo don't look o_O
iris = case unsafePerformIO $ loadIris "iris.csv" of
  Right x -> V.toList x
  _ -> error "Oops!"