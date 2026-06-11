# GridLock 2.0 — Our Approach

## What problem we were solving

The task is to predict **traffic demand** for many small map locations during the
daytime of day 49. Each row in the data is one location (given as a *geohash*) at
one 15‑minute time slot, plus some details about that road — its type, how many
lanes it has, whether large vehicles are allowed, whether there is a landmark
nearby, the temperature, and the weather. Demand is a number between 0 and 1.

The training data gives us a full picture of day 48 (the whole day) and the very
early hours of day 49. The test set asks us to fill in the daytime hours of day
49. So in plain terms: *we have seen how each place behaves over a full day, and
we need to predict how those same places behave the next day during the hours we
were not shown.*

## How we understood the data

We spent our first effort just looking at the numbers, because the patterns in
this kind of traffic problem are usually simple and physical:

1. **Location is by far the most important thing.** When we grouped the data by
   geohash and looked at the average demand for each place, that average alone
   lined up very strongly with the real demand (correlation around 0.83). This
   makes sense — a busy highway junction is busy most of the time, and a quiet
   residential lane stays quiet. Almost all of the test locations (1,180 out of
   1,190) also appear in the training data, so the history of each place is
   genuinely useful.

2. **Road type matters a lot.** Highways average about 0.61 demand, streets about
