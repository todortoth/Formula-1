# Formula 1 Strategic Decisions

Based on FastF1 API

## Project Vision

> „Formula 1 is the Pinnacle of Motorsports.”
> — _Fédération Internationale de l'Automobile_

**Formula 1 Strategic Decisions** is an advanced analytics and graph-based modeling tool
designed to simulate, visualize and eventually **predict optimal race strategies** (such as undercut/overcut windows, etc.)

By transforming raw session telemetry into relational graph structures the tool captures complex spatial relationships, 
such as driver proximity windows, track status changes (Safety Car/VSC), and sector-by-sector time gaps, to feed into future 
machine learning and tactical decision making models.

## Tech stack & Dependencies

- Pyhton 3.10+
- FastF1
- Pandas, Numpy

## Project Architecture

```
├── main.py                   # Application entry point & console/graph output runner
├── SessionAnalyserClass.py   # Core processor for FastF1 data, snapshots, & graph building
├── DriverNodeClass.py        # Data class capturing individual driver telemetry states
├── SectorNode.py             # Class handling track status conditions (SC, VSC, Flags)
```

## Future Roadmap

[ ] **Predictive Strategy Engine**: Train machine learning models on historical race graphs 
to recommend optimal pit stops and tyre compounds in real-time.
