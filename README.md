# Dynamic Mode Decomposition (DMD)

https://en.wikipedia.org/wiki/Dynamic_mode_decomposition

This repository is my sandbox for experimenting with DMD.

1. Working on a streaming (incremental) DMD algorithm

2. Experimenting with the pydmd package (very well done!) and possibly
implementing a sliding window approach to compare with an incremental
approach.

Notes to self:

Switching the POD compression (step 3) to after the update step (4)
makes more sense logically to me and produces more sensible results in
practice.  Otherwise we are expanding the internal structures, doing
very little, and compressing them again.  Basically we stop updating
the zero frequency mode once the POD compression kicks in.

The zero frequency mode is effectively some scaled average of the
frames seen.  Would be much faster to just average frames if we don't
need the higher frequencey modes for anything.

In the "forgetting" version of the algorithm, we can't reconstruct the
scene because the A(:,1) (first input column) is forgotten and what is
the new starting basis now? But do we need to reconstruct the scene?
Probably not.

Is DMD really the best tool for this, or just a cool way to apply
interseting technology that produces a useful result, but via much
more computational effort with a lot of extra things computed that
aren't ever used or needed?

How would DMD handle blowing leaves or branches?  Flags?  Changing
shadows/light? Clouds passing over, etc.

DMD just computes a set of modes that fit/model the behavior of the
inputs, but each input (in our case) is just a pixel and these aren't
grouped or linked in anyway ... other than our human eyes/brains can
see the groupings in the mode plots.

  - Does it make sense to look for higher level structures
    explicitely?  edge detection, line fitting, feature detection?  We
    could start to track meaning from one frame to the next.

I feel like the paper "Multi-Sensor Scene Segmentation for Unmanned
Air and Ground Vehicles using Dynamic Mode Decomposition" is saying
they are showing results from the streaming algorithm, but they are
really showing the full solution (or the streaming with no POD
compression.)  They show "frame 6" but the background shows the bike
passing along the whole trajectory ... so they have clearly run the
data to completion before generating the picture and aren't showing
the state of the system as of the state frame.  The reason this is
important is because the algorithm doesn't do what they claim or show
as it's implemented, so they clearly used a bit of slight of hand to
get the result without fixing the algorithm.