%include polycode.fmt

%% ── Format directives ────────────────────────────────────────────────────────
%% Symbolic operators rendered as math
%format ->          = "\to"
%format =>          = "\Rightarrow"
%format forall      = "\forall"
%format <$>         = "\mathbin{\langle\$\rangle}"
%format `div`       = "\mathbin{\div}"

%% Named variables → Greek / math letters (use these names in code blocks)
%format phi         = "\phi"
%format alpha       = "\alpha"

%% Lens type operators
%format Lens'       = "\mathit{Lens}^{\prime}"
%% ─────────────────────────────────────────────────────────────────────────────

\section{Lenses and Optics}
\label{sec:lenses}

Compound data structures are ubiquitous in modern programming, and their
inherent modularity is key to building and reasoning with complex ensembles.
Lenses\cite{foster2005combinators} are one approach to solve the problem of
accessing the components of such structures, also known as the
\textit{view-update problem}\cite{updatesemantics}.

\subsection{Origins}

A simple lens from outer type $S$ to inner type $A$ (which we shall denote as
|Lens' s a| for reasons which will become apparent) can be thought of as
comprising\cite{foster2005combinators} a function |view :: s -> a| which
projects the component $A$, and a function |update :: s -> a -> s| which
accepts a new value of $A$ and uses it to modify a given $S$.

\begin{code}
data Lens' s a = Lens'
    {  view    :: s -> a
    ,  update  :: s -> a -> s
    }
\end{code}

As a concrete example, consider the relationship between the following
|Account| and |PhoneNumber| data types:

\begin{code}
newtype PhoneNumber  = PhoneNumber Int
newtype Account      = Account
    {  phone      :: PhoneNumber
    ,  accountId  :: Int
    }

acctToPhone :: Lens' Account PhoneNumber
acctToPhone = Lens' view update
  where
    view    :: Account -> PhoneNumber
    view    = phone

    update  :: Account -> PhoneNumber -> Account
    update acct num = acct { phone = num }
\end{code}

\subsection{Van Laarhoven Lenses}

It is tempting to extend our |Lens| type with a modification function
|mod :: (a -> a) -> s -> s|\cite{peytonjones2013lenses}. This begs the
question: at what point do we stop? We could usefully include
|modM :: (a -> Maybe a) -> s -> Maybe s|, or even
|modIO :: (a -> IO a) -> s -> IO s|. Before letting this get out of hand, we
observe that both share the structure of a |Functor|, yielding
|modF :: Functor f => (a -> f a) -> s -> f s|.

As a quick refresher, the category of |Functor|s\cite{haskell2010,
haskell-base-functor} supports the lifting of an arbitrary function into their
type. In Haskell terms:

\begin{code}
class Functor f where
    fmap :: (a -> b) -> f a -> f b
\end{code}

By careful choice of specific |Functor|s we can recover |view|, |mod|, and
|update|. Note that providing |const a| to |mod| recovers |update|, so we need
only produce the first two. The key choices are:

\begin{code}
newtype Identity  a    = Identity  {  runIdentity  :: a    }
newtype Const     a b  = Const     {  getConst     :: a    }

instance Functor Identity where
    fmap f (Identity x) = Identity (f x)

instance Functor (Const m) where
    fmap _ (Const v) = Const v
\end{code}\cite{ghc_internal_identity, ghc_internal_const}

Applying |Identity| to |modF| yields |modF :: (a -> Identity a) -> s -> Identity s|,
isomorphic to |mod|---all that is left is wrapping and unwrapping the
|Identity|\footnote{This has no runtime cost thanks to the semantics of
\texttt{newtype} and \texttt{coerce}\cite{breitner2014safe}.}.

Applying |Const a| to |modF| yields
|modF :: (a -> Const a a) -> s -> Const a s|,
isomorphic to |(a -> a) -> s -> a|, which when provided |id| is exactly |view|.

Thus a |Lens'| only needs |modF|, and we can simplify to the type alias:

\begin{code}
type Lens' s a = forall f . Functor f => (a -> f a) -> s -> f s
\end{code}

This representation also lets GHC apply aggressive optimisations for
higher-order functions, which will matter as we compose increasingly complex
lenses.

It is often convenient to allow the types in the reverse direction to differ
from those in the forward direction, yielding
|update :: s -> b -> t| for fresh types $B$ and $T$:

\begin{code}
data Lens s t a b = Lens
    {  view    :: s -> a
    ,  update  :: s -> b -> t
    }
\end{code}

In the Van Laarhoven style:

\begin{code}
type Lens s t a b  = forall f . Functor f => (a -> f b) -> s -> f t
type Lens' s a     = Lens s s a a

lens :: (s -> a) -> (s -> b -> t) -> Lens s t a b
-- Note: fmap written infix
lens sa sbt afb s = sbt s <$> afb (sa s)
\end{code}\cite{ekmett2025lens}

\subsection{Composition and Modularity}

The Van Laarhoven encoding gives composition for free. Given:

\begin{code}
x  :: Lens s t a b  -- forall f . Functor f => (a -> f b) -> (s -> f t)
y  :: Lens a b c d  -- forall f . Functor f => (c -> f d) -> (a -> f b)
\end{code}

Standard function composition |(.)| has type |(b -> c) -> (a -> b) -> (a -> c)|,
so |x . y| unifies immediately:

\begin{code}
-- x . y :: Lens s t c d
--        = forall f . Functor f => (c -> f d) -> (s -> f t)
\end{code}

No glue code required: composing lenses with |(.)| is simply function
composition. This allows compound accessors to be built from primitives without
any overhead.

\subsection{Final Note}

So far we have considered only lenses whose focus is a concrete field of a
record, as with |PhoneNumber| inside |Account|. This is not a requirement, and
in subsequent sections we will see that machine learning layers can themselves
be expressed as more exotic lenses.
