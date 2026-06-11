%include polycode.fmt

%% ── Format directives ────────────────────────────────────────────────────────
%format ->    = "\to"
%format =>    = "\Rightarrow"
%format forall = "\forall"
%format .#.   = "\mathbin{\bullet}"
%format ***   = "\mathbin{\times}"
%format <$>   = "\mathbin{\langle\$\rangle}"
%% ─────────────────────────────────────────────────────────────────────────────

%if False
\begin{code}
module Design where

import Core
import Control.Lens
import Control.Arrow ((***))
\end{code}
%endif

\chapter{Core Implementation}
\label{chap:core}

\section{From Lenses to Parametric Lenses}

Building on the Van Laarhoven lenses of the previous chapter, we now introduce
\emph{parametric lenses}~\cite{catlearning}: an extension that promotes the
parameter of a computation to a first-class wire in the type.  A standard lens
|Lens s t a b| is stateless---it carries no mutable configuration of its own.
For machine learning this is insufficient: a neural network layer holds weight
matrices and bias vectors that are read during the forward pass and updated
during the backward pass.  Parametric lenses make these parameters structural,
so that the update rule is part of the type rather than an afterthought.

\section{Wire Diagrams}
\label{sec:wirediagrams}

A lens |Lens A A' B B'| has a direct operational reading as two functions: a
\emph{forward} computation $f : A \to B$ and a \emph{backward} computation
$f^* : A \times B' \to A'$.  The backward pass requires the original input
$A$, so the wire carrying $A$ branches: one copy feeds $f$ while the other
curves to the right-hand side of $f^*$.

\begin{figure}[h]
\centering
\begin{tikzpicture}
  \draw[thick] (-3,-1.2) rectangle (3,1.2);
  \node[draw, minimum width=1.4cm, minimum height=0.65cm] (f)  at (0,  0.7) {$f$};
  \node[draw, minimum width=1.4cm, minimum height=0.65cm] (fs) at (0, -0.7) {$f^*$};
  \coordinate (fork) at (-1.5, 0.7);
  %% A enters and branches silently at fork
  \draw[wire] (-4.2,  0.7) -- node[above] {$A$}  (-3, 0.7);
  \draw[thick] (-3, 0.7) -- (fork);
  \draw[wire] (fork) -- (f.west);
  %% memorised branch: two-segment arc; first sweep clears f, second gives soft landing
  \draw[wire] (fork) .. controls (-0.8, 0.0) and (0.9, 0.1) .. (1.2,-0.1)
              .. controls (1.35,-0.2) and (1.0,-0.38) .. ($(fs.east)+(0,0.2)$);
  %% f -> B
  \draw[wire] (f.east) -- (3, 0.7);
  \draw[wire] (3,  0.7) -- node[above] {$B$}  (4.2, 0.7);
  %% B' -> f* (enters right side)
  \draw[wire] (4.2, -0.7) -- node[below] {$B'$} (3, -0.7);
  \draw[wire] (3, -0.7) -- (fs.east);
  %% f* -> A'
  \draw[wire] (fs.west) -- (-3, -0.7);
  \draw[wire] (-3,-0.7) -- node[below] {$A'$} (-4.2, -0.7);
\end{tikzpicture}
\caption{$\mathit{Lens}\;A\;A'\;B\;B'$ decomposed into forward pass $f$ and
  backward pass $f^*$.}
\label{fig:lens-ff}
\end{figure}

Substituting $A = (p, a)$ and $A' = (p', a')$ gives the same diagram for a
|Lens (p,a) (p',a') b b'|, in which the parameter $p$ is bundled with the
data on the horizontal wire:

\begin{figure}[h]
\centering
\begin{tikzpicture}
  \draw[thick] (-3,-1.2) rectangle (3,1.2);
  \node[draw, minimum width=1.4cm, minimum height=0.65cm] (f)  at (0,  0.7) {$f$};
  \node[draw, minimum width=1.4cm, minimum height=0.65cm] (fs) at (0, -0.7) {$f^*$};
  \coordinate (fork) at (-1.5, 0.7);
  \draw[wire] (-4.2,  0.7) -- node[above] {$(p,a)$}   (-3, 0.7);
  \draw[thick] (-3,  0.7) -- (fork);
  \draw[wire] (fork) -- (f.west);
  \draw[wire] (fork) .. controls (-0.8, 0.0) and (0.9, 0.1) .. (1.2,-0.1)
              .. controls (1.35,-0.2) and (1.0,-0.38) .. ($(fs.east)+(0,0.2)$);
  \draw[wire] (f.east) -- (3, 0.7);
  \draw[wire] (3,  0.7) -- node[above] {$b$}   (4.2, 0.7);
  \draw[wire] (4.2, -0.7) -- node[below] {$b'$}  (3, -0.7);
  \draw[wire] (3, -0.7) -- (fs.east);
  \draw[wire] (fs.west) -- (-3, -0.7);
  \draw[wire] (-3,-0.7) -- node[below] {$(p',a')$} (-4.2, -0.7);
\end{tikzpicture}
\caption{Substituting $A=(p,a)$, $A'=(p',a')$: parameter and data bundled on
  the horizontal wire.}
\label{fig:lens-para-tuple}
\end{figure}

Separating the $p$ component onto a dedicated \emph{vertical} axis---$p$
entering from above on the left and $p'$ exiting upward on the right---and
retaining only $a$, $a'$ on the horizontal data wires yields the canonical
wire diagram for a parametric lens:

\begin{figure}[h]
\centering
\begin{tikzpicture}
  \draw[thick] (-3,-1.4) rectangle (3,1.8);
  \node[draw, minimum width=1.4cm, minimum height=0.65cm] (f)  at (-0.5,  0.7) {$f$};
  \node[draw, minimum width=1.4cm, minimum height=0.65cm] (fs) at ( 0.5, -0.7) {$f^*$};
  \coordinate (fork) at (-2.0, 0.7);
  %% p' drawn first so arc and b cross over it
  \draw[wire] (fs.north) -- node[right] {$p'$} (0.5, 2.8);
  %% data wires
  \draw[wire] (-4.2,  0.7) -- node[above] {$a$}  (-3, 0.7);
  \draw[thick] (-3, 0.7) -- (fork);
  \draw[wire] (fork) -- (f.west);
  \draw[wire] (f.east) -- (3, 0.7);
  \draw[wire] (3,  0.7) -- node[above] {$b$}  (4.2, 0.7);
  \draw[wire] (4.2, -0.7) -- node[below] {$b'$} (3, -0.7);
  \draw[wire] (3, -0.7) -- (fs.east);
  \draw[wire] (fs.west) -- (-3, -0.7);
  \draw[wire] (-3,-0.7) -- node[below] {$a'$} (-4.2, -0.7);
  %% memorised branch: fork curves to f*.east
  \draw[wire] (fork) .. controls (-1.5, 0.1) and (-0.8,-0.1) .. (-0.4,-0.15)
              .. controls (0.2,-0.2) and (0.9,-0.15) .. (1.0,-0.15)
              .. controls (2.6,-0.2) and (1.5,-0.45) .. ($(fs.east)+(0,0.2)$);
  %% p wire
  \draw[wire] (-0.5, 2.8) -- node[left] {$p$} (f.north);
\end{tikzpicture}
\caption{|ParaLens p p' a a' b b'|: $p$ and $p'$ factored onto the vertical axis.}
\label{fig:paralens-single}
\end{figure}

The left and right edges carry the data wires ($a$ and $b$ on the upper
forward channels, $a'$ and $b'$ on the lower backward channels), and the top
edge carries both parameter wires.  Setting the Van Laarhoven functor to
|Identity| recovers the forward pass $(p,a) \to b$; setting it to |Const b'|
recovers the backward pass $(p,a,b') \to (p',a')$.  The design therefore
inherits lens composition, the lens library, and all compiler optimisations
for free.

\section{The |ParaLens| Type}

The implementation realises the wire-diagram intuition as a type alias over the
standard Van Laarhoven lens:

\begin{code}
type ParaLens p p' a a' b b' = Lens (p,a) (p',a') b b'
\end{code}

The six type variables have the following roles:

\begin{center}
\renewcommand{\arraystretch}{1.3}
\begin{tabular}{cl}
  \toprule
  Variable & Role \\
  \midrule
  |p|, |p'|   & parameter type before and after the update \\
  |a|, |a'|   & data type in the forward and backward directions \\
  |b|, |b'|   & output type in the forward and backward directions \\
  \bottomrule
\end{tabular}
\end{center}

Expanding the alias, |ParaLens p p' a a' b b'| is:
\[
  \forall f.\; \mathit{Functor}\; f \Rightarrow
  \bigl((p,a) \to f\; b\bigr) \to (p,a) \to f\;(p',a')
\]
Setting |f = Identity| recovers the forward pass $f$; setting
|f = Const b'| recovers the backward pass $f^*$, exactly as with
ordinary lenses.

The simplified variant |ParaLens'| fixes the types to be unchanged by the
pass---the natural choice for layers that do not change their own interface:

\begin{code}
type ParaLens' p a b = ParaLens p p a a b b
\end{code}

\section{Lifting Ordinary Lenses}

A standard lens carries no parameters at all. It can be embedded into the
parametric setting by equipping it with the unit parameter |()| via |toPara|:

\begin{code}
toPara :: Lens a a' b b' -> ParaLens () () a a' b b'
toPara = (leftUnit .)
\end{code}

The helper |leftUnit| is the isomorphism witnessing that |((), a)| is
isomorphic to |a| --- pairing with the unit type adds no information:

\begin{code}
leftUnit :: Iso ((),a) ((),a') a a'
leftUnit = iso snd ((),)
\end{code}

\noindent Here |Iso s t a b| is the van~Laarhoven type for an isomorphism
between |(s, t)| and |(a, b)|, exported by \texttt{Control.Lens}~\cite{ekmett2025lens};
|iso :: (s -> a) -> (b -> t) -> Iso s t a b| constructs one from a pair of
inverse functions.  |leftUnit| is built from |snd :: ((), a) -> a| (the
forward direction) and |((),) :: a -> ((), a)| (the backward direction).

Composing |leftUnit| on the left of any lens rearranges the source pair so
that the trivial parameter slot disappears from the type. Symmetrically,
|rightUnit| handles the case where the unit appears on the right:

\begin{code}
rightUnit :: Iso (a,()) (a',()) a a'
rightUnit = iso fst (,())
\end{code}

\section{The Monoidal Product}

Composition is not the only structural operation. Given a lens that acts on
some component, it is frequently necessary to \emph{extend} it to act on one
component of a pair while leaving the other untouched. This is the role of
|rightLens| and |leftLens|:

\begin{code}
rightLens :: Lens p p' q q' -> Lens (a,p) (a,p') (a,q) (a,q')
rightLens = alongside id

leftLens  :: Lens p p' q q' -> Lens (p,a) (p',a) (q,a) (q',a)
leftLens  = flip alongside id
\end{code}

These are the two injections of the monoidal product: |alongside| from
\texttt{Control.Lens} pairs two lenses into a lens on a product type. Passing
|id| on one side leaves that component untouched. In wire-diagram terms,
|rightLens l| adds an \emph{extra wire on the left} that passes through the
box unchanged:

\begin{figure}[h]
\centering
\begin{tikzpicture}
  %% Outer box (rightLens l as a whole)
  %% Symmetric 0.4 gaps: top(1.8)-a(1.4)-a'(1.0)-[id/l gap]-p(0.2)-p'(-0.2)-bottom(-0.6)
  \draw[thick] (-2.5, -0.6) rectangle (2.5, 1.8);

  %% l box (lower section)
  \node[paralens, minimum width=1.4cm, minimum height=0.9cm] (box) at (0, 0) {$l$};

  %% id box (upper section, dotted): centre 1.2, half-height 0.45 → spans [0.75, 1.65]
  %% wires at 1.2±0.2 = 1.4 and 1.0, both well inside the box with label room to spare
  \node[draw, dotted, thick, minimum width=1.4cm, minimum height=0.9cm] (id) at (0, 1.2) {};

  %% a forward: labeled external stub, straight through id, labeled external stub
  \draw[wire] (-4.0,  1.4) -- node[above] {$a$}  (-2.5,  1.4);
  \draw[wire] (-2.5,  1.4) -- ( 2.5,  1.4);
  \draw[wire] ( 2.5,  1.4) -- node[above] {$a$} ( 4.0,  1.4);

  %% a' backward: labels shifted outward (pos=0.25) to clear the p/q labels below
  \draw[wire] ( 4.0,  1.0) -- node[below, pos=0.35] {$a'$} ( 2.5,  1.0);
  \draw[wire] ( 2.5,  1.0) -- (-2.5,  1.0);
  \draw[wire] (-2.5,  1.0) -- node[below, pos=0.65] {$a'$} (-4.0,  1.0);

  %% p: label shifted inward (pos=0.75) to clear the a' label above
  \draw[wire] (-4.0,  0.2) -- node[above, pos=0.65] {$p$}  (-2.5,  0.2);
  \draw[wire] (-2.5,  0.2) -- ($(box.west)+(0,  0.2)$);

  %% p': internal from l, labeled external stub
  \draw[wire] ($(box.west)+(0, -0.2)$) --           (-2.5, -0.2);
  \draw[wire] (-2.5, -0.2) -- node[below] {$p'$} (-4.0, -0.2);

  %% q: label shifted inward (pos=0.25) to clear the a' label above
  \draw[wire] ($(box.east)+(0,  0.2)$) --           ( 2.5,  0.2);
  \draw[wire] ( 2.5,  0.2) -- node[above, pos=0.35] {$q$}  ( 4.0,  0.2);

  %% q': labeled external stub, internal to l
  \draw[wire] ( 4.0, -0.2) -- node[below] {$q'$} ( 2.5, -0.2);
  \draw[wire] ( 2.5, -0.2) -- ($(box.east)+(0, -0.2)$);
\end{tikzpicture}
\caption{|rightLens l| (outer box): $\mathrm{id}$ passes $a$ and $a'$
  through unchanged, while $l$ transforms $p \to q$ forward and $q' \to p'$ backward.}
\label{fig:rightlens}
\end{figure}

\section{Composition}
\label{sec:paralens-composition}

The central operation is sequential composition, written |(.#.)|.  Given two
parametric lenses whose data types are compatible:

\begin{code}
f  :: ParaLens p  p'  a  a'  b  b'
g  :: ParaLens q  q'  b  b'  c  c'
\end{code}

\noindent their composition |f .#. g| threads the output of |f| into the
input of |g|, and \emph{pairs} the two parameter wires into a single product
wire:

\begin{figure}[h]
\centering
\begin{tikzpicture}
  \node[paralens] (f) {$f$};
  \node[paralens, right=2.8cm of f] (g) {$g$};

  %% a/a' on left of f
  \draw[wire] ($(f.west)+(-1.8,  0.25)$) -- node[above] {$a$}  ($(f.west)+(0, 0.25)$);
  \draw[wire] ($(f.west)+(0,    -0.25)$) -- node[below] {$a'$} ($(f.west)+(-1.8,-0.25)$);

  %% b/b' between f and g (forward upper, backward lower)
  \draw[wire] ($(f.east)+(0, 0.25)$) -- node[above] {$b$}  ($(g.west)+(0, 0.25)$);
  \draw[wire] ($(g.west)+(0,-0.25)$) -- node[below] {$b'$} ($(f.east)+(0,-0.25)$);

  %% c/c' on right of g
  \draw[wire] ($(g.east)+(0, 0.25)$)     -- node[above] {$c$}  ($(g.east)+(1.8, 0.25)$);
  \draw[wire] ($(g.east)+(1.8,-0.25)$)   -- node[below] {$c'$} ($(g.east)+(0,  -0.25)$);

  %% p/p' on f: both at top edge, p down-into, p' up-out
  \draw[wire] ($(f.north)+(-0.3, 1.2)$) -- node[left]  {$p$}  ($(f.north)+(-0.3, 0)$);
  \draw[wire] ($(f.north)+( 0.3, 0)$)   -- node[right] {$p'$} ($(f.north)+( 0.3, 1.2)$);

  %% q/q' on g
  \draw[wire] ($(g.north)+(-0.3, 1.2)$) -- node[left]  {$q$}  ($(g.north)+(-0.3, 0)$);
  \draw[wire] ($(g.north)+( 0.3, 0)$)   -- node[right] {$q'$} ($(g.north)+( 0.3, 1.2)$);
\end{tikzpicture}
\caption{Composition |f .#. g :: ParaLens (p,q) (p',q') a a' c c'|. The
  bidirectional middle channel shows $b$ (forward, upper) and $b'$ (backward,
  lower) flowing in opposite directions between the two layers.}
\label{fig:composition}
\end{figure}

The result has type |ParaLens (p,q) (p',q') a a' c c'|: the combined
parameter is the \emph{product} of the two individual parameter types.  This
directly models backpropagation: during the backward pass, gradients flow
through |g| and then |f|, with each layer independently updating its own
slice of the combined parameter.

\begin{figure}[h]
\centering
\begin{tikzpicture}
  \draw[thick] (-5.5,-1.4) rectangle (5.5,2.0);
  \node[draw, minimum width=1.4cm, minimum height=0.65cm] (f)  at (-3.5,  0.7) {$f$};
  \node[draw, minimum width=1.4cm, minimum height=0.65cm] (fs) at (-2.5, -0.7) {$f^*$};
  \node[draw, minimum width=1.4cm, minimum height=0.65cm] (g)  at ( 3.5,  0.7) {$g$};
  \node[draw, minimum width=1.4cm, minimum height=0.65cm] (gs) at ( 2.5, -0.7) {$g^*$};
  %% p' and q' drawn first so b wire crosses over them
  \draw[wire] (fs.north) -- node[right] {$p'$} (-2.5, 2.8);
  \draw[wire] (gs.north) -- node[right] {$q'$} ( 2.5, 2.8);
  %% external data wires
  \draw[wire] (-6.8,  0.7) -- node[above] {$a$}  (-5.5,  0.7);
  \draw[wire] (-5.5,  0.7) -- (f.west);
  \draw[wire] (g.east) -- (5.5,  0.7);
  \draw[wire] (5.5,   0.7) -- node[above] {$c$}  (6.8,  0.7);
  \draw[wire] (6.8,  -0.7) -- node[below] {$c'$} (5.5, -0.7);
  \draw[wire] (5.5,  -0.7) -- (gs.east);
  \draw[wire] (fs.west) -- (-5.5, -0.7);
  \draw[wire] (-5.5, -0.7) -- node[below] {$a'$} (-6.8, -0.7);
  %% internal b/b' connecting the two sub-lenses
  \draw[wire] (f.east)  -- node[above] {$b$}  (g.west);
  \draw[wire] (gs.west) -- node[below] {$b'$} (fs.east);
  %% p and q parameter wires
  \draw[wire] (-3.5, 2.8) -- node[left]  {$p$} (f.north);
  \draw[wire] ( 3.5, 2.8) -- node[right] {$q$} (g.north);
\end{tikzpicture}
\caption{|f .#. g| as a single |ParaLens (p,q) (p',q') a a' c c'|: each
  sub-lens carries its own independent parameter channel on the vertical axis,
  and the intermediate wire $b$/$b'$ links the two halves.}
\label{fig:composition-box}
\end{figure}

The implementation uses three auxiliary isomorphisms to reassemble the nested
pairs into the required shape:

\begin{code}
infixr 8 .#.
(.#.)  ::  ParaLens p  p'  a  a'  b  b'
       ->  ParaLens q  q'  b  b'  c  c'
       ->  ParaLens (p,q) (p',q') a  a'  c  c'
f .#. g = swapFst . rotate . rightLens f . g
\end{code}

Reading right-to-left, and tracking the source type at each step:

\begin{enumerate}

\item |g :: Lens (q,b) (q',b') c c'|.  This is the inner parametric lens mapping |b| to |c| while carrying |q| as a parameter.

\item |rightLens f| extends |f :: Lens (p,a) (p',a') b b'| to act on the
  right slot of a pair with |q| on the left.  Composed with |g|:
  \[
    \mathit{rightLens}\; f \mathbin{\circ} g
    \;::\; \mathit{Lens}\;(q,(p,a))\;(q',(p',a'))\;c\;c'
  \]

\item |rotate :: Iso ((a,b),c) _ (a,(b,c)) _| reassociates the nested triple,
  changing the source from |(q,(p,a))| to |((q,p),a)|.

\item |swapFst :: Iso ((a,b),c) _ ((b,a),c) _| swaps the inner pair, changing
  |((q,p),a)| to |((p,q),a)|.

\end{enumerate}

The final source type is |((p,q),a)|, which matches |ParaLens (p,q) (p',q') a a' c c'|
as required.

\section{Reparametrisation}
\label{sec:repara}

Given a lens |r :: Lens q q' p p'| that relates two parameter spaces, we can
\emph{reparametrise} any |ParaLens p p' a a' b b'| to run in the |q| parameter
space:

\begin{code}
repara  ::  Lens q q' p p'
        ->  ParaLens p  p'  a  a'  b  b'
        ->  ParaLens q  q'  a  a'  b  b'
repara r = (leftLens r .)
\end{code}

This is the mechanism by which an \emph{optimiser}---itself a lens that maps
raw gradients to updated parameters---can be wired into a model without
changing the model's own structure. Composing |leftLens r| on the left
redirects the parameter wires through |r| before they reach the model, so the
model continues to ``see'' its own parameter type while the outer context
handles the optimiser state.

% \section{Lifting to Batches}

% The function |liftUpdate| lifts a lens on a single parameter |p| to a lens on
% a \emph{list} of parameters |[p]|, enabling batch processing:

% \begin{code}
% liftUpdate :: Lens' p p -> Lens' [p] [p]
% liftUpdate ul = lens (map (view ul)) (\ps gs -> zipWith (set ul) gs ps)
% \end{code}

% The forward direction maps |view ul| over the list, extracting the focused
% component from each parameter. The backward direction uses |set ul| to write
% each updated gradient back into the corresponding parameter via |zipWith|.
% This allows a single-element parametric lens to be replicated across a batch
% without any changes to the lens itself.

\section{Efficiency Considerations}
At first glance, the |ParaLens| type appears to introduce a lot of extra
pairing and unpairing of parameters, which could be a source of overhead. However,
the implementation relies on the fact that these operations are \emph{isomorphisms}
that can be optimised away by the compiler. The |swapFst|, |rotate|, and |rightLens| 
functions are all built from simple tuple manipulations that the Haskell compiler can 
inline and eliminate, resulting in no runtime overhead for the composition operation. 
This means that the elegant compositional structure of |ParaLens| does not
come at the cost of performance, and the abstraction can be used freely
without worrying about efficiency penalties.
