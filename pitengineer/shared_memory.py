"""Read Assetto Corsa live telemetry via the Windows Shared Memory API.

AC publishes three memory-mapped files while running:
    Local\\acpmf_physics   - real-time physics (tyre temps, slip, inputs, speed)
    Local\\acpmf_graphics  - session/lap state (lap times, laps done, in-pit)
    Local\\acpmf_static    - constants (car model, track, max rpm)

We mirror the C structs with ctypes and read them each tick. Only the leading
fields we actually use are declared for physics/graphics; because struct fields
are sequential, offsets stay correct as long as the declared prefix matches
AC's layout. The live monitor (monitor.py) is the sanity check: if speed and
tyre temps read as plausible numbers, the layout is right.

Windows only (memory-mapped shared memory). Import guarded so the rest of the
package still works on other platforms / without AC.
"""

from __future__ import annotations

import ctypes
from ctypes import c_float, c_int, c_wchar
from dataclasses import dataclass

# Candidate names for each block, tried in order. AC names its shared-memory
# objects differently across builds/namespaces, so we try the bare name and the
# session-local variant and use whichever opens.
PHYSICS_TAGS = ("acpmf_physics", "Local\\acpmf_physics")
GRAPHICS_TAGS = ("acpmf_graphics", "Local\\acpmf_graphics")
STATIC_TAGS = ("acpmf_static", "Local\\acpmf_static")

# Win32 constants for OpenFileMappingW / MapViewOfFile.
_FILE_MAP_READ = 0x0004


class _Physics(ctypes.Structure):
    _fields_ = [
        ("packetId", c_int),
        ("gas", c_float),
        ("brake", c_float),
        ("fuel", c_float),
        ("gear", c_int),
        ("rpms", c_int),
        ("steerAngle", c_float),
        ("speedKmh", c_float),
        ("velocity", c_float * 3),
        ("accG", c_float * 3),
        ("wheelSlip", c_float * 4),
        ("wheelLoad", c_float * 4),
        ("wheelsPressure", c_float * 4),
        ("wheelAngularSpeed", c_float * 4),
        ("tyreWear", c_float * 4),
        ("tyreDirtyLevel", c_float * 4),
        ("tyreCoreTemperature", c_float * 4),
        ("camberRAD", c_float * 4),
        ("suspensionTravel", c_float * 4),
        ("drs", c_float),
        ("tc", c_float),
        ("heading", c_float),
        ("pitch", c_float),
        ("roll", c_float),
        ("cgHeight", c_float),
        ("carDamage", c_float * 5),
        ("numberOfTyresOut", c_int),
        ("pitLimiterOn", c_int),
        ("abs", c_float),
    ]


class _Graphics(ctypes.Structure):
    _fields_ = [
        ("packetId", c_int),
        ("status", c_int),          # 0 OFF, 1 REPLAY, 2 LIVE, 3 PAUSE
        ("session", c_int),
        ("currentTime", c_wchar * 15),
        ("lastTime", c_wchar * 15),
        ("bestTime", c_wchar * 15),
        ("split", c_wchar * 15),
        ("completedLaps", c_int),
        ("position", c_int),
        ("iCurrentTime", c_int),    # ms
        ("iLastTime", c_int),       # ms
        ("iBestTime", c_int),       # ms
        ("sessionTimeLeft", c_float),
        ("distanceTraveled", c_float),
        ("isInPit", c_int),
        ("currentSectorIndex", c_int),
        ("lastSectorTime", c_int),
        ("numberOfLaps", c_int),
        ("tyreCompound", c_wchar * 33),
        ("replayTimeMultiplier", c_float),
        ("normalizedCarPosition", c_float),  # 0..1 lap fraction - where on track
    ]


class _Static(ctypes.Structure):
    _fields_ = [
        ("smVersion", c_wchar * 15),
        ("acVersion", c_wchar * 15),
        ("numberOfSessions", c_int),
        ("numCars", c_int),
        ("carModel", c_wchar * 33),
        ("track", c_wchar * 33),
        ("playerName", c_wchar * 33),
        ("playerSurname", c_wchar * 33),
        ("playerNick", c_wchar * 33),
        ("sectorCount", c_int),
        ("maxTorque", c_float),
        ("maxPower", c_float),
        ("maxRpm", c_int),
        ("maxFuel", c_float),
        ("suspensionMaxTravel", c_float * 4),
        ("tyreRadius", c_float * 4),
    ]


# --- Plain-data snapshots we hand to the rest of the app -------------------

# AC_STATUS meanings for graphics.status
STATUS = {0: "OFF", 1: "REPLAY", 2: "LIVE", 3: "PAUSE"}


@dataclass
class PhysicsSnapshot:
    gas: float
    brake: float
    steer: float
    speed_kmh: float
    gear: int
    rpm: int
    # Per-wheel, order: [FL, FR, RL, RR]
    tyre_core_temp: tuple[float, float, float, float]
    tyre_pressure: tuple[float, float, float, float]
    wheel_slip: tuple[float, float, float, float]
    wheel_load: tuple[float, float, float, float]
    suspension_travel: tuple[float, float, float, float]


@dataclass
class GraphicsSnapshot:
    status: str
    completed_laps: int
    current_time_ms: int
    last_time_ms: int
    best_time_ms: int
    is_in_pit: bool
    current_sector: int
    tyre_compound: str
    car_position: float  # normalized 0..1 lap fraction (where on the track)


@dataclass
class StaticSnapshot:
    car_model: str
    track: str
    max_rpm: int


def _quad(arr) -> tuple[float, float, float, float]:
    return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))


class _MappedBlock:
    """One shared-memory block opened existing-only via Win32.

    Uses OpenFileMappingW so it FAILS when AC isn't running (unlike mmap with a
    tagname, which would create a new empty block and mask the fact the game is
    off). Kept mapped; each read copies a fresh struct from the view.
    """

    def __init__(self, tags: tuple[str, ...], struct_cls) -> None:
        self._k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._k32.OpenFileMappingW.restype = ctypes.c_void_p
        self._k32.OpenFileMappingW.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p]
        self._k32.MapViewOfFile.restype = ctypes.c_void_p
        self._k32.MapViewOfFile.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_size_t]
        self._k32.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
        self._k32.CloseHandle.argtypes = [ctypes.c_void_p]

        self._struct_cls = struct_cls
        self._size = ctypes.sizeof(struct_cls)
        self._handle = None
        self._view = None

        for tag in tags:
            handle = self._k32.OpenFileMappingW(_FILE_MAP_READ, False, tag)
            if handle:
                view = self._k32.MapViewOfFile(handle, _FILE_MAP_READ, 0, 0, self._size)
                if view:
                    self._handle, self._view = handle, view
                    return
                self._k32.CloseHandle(handle)
        raise FileNotFoundError(
            f"None of the shared-memory names {tags} could be opened. "
            "Is Assetto Corsa running and in a session?"
        )

    def read(self):
        return self._struct_cls.from_address(self._view)

    def close(self) -> None:
        try:
            if self._view:
                self._k32.UnmapViewOfFile(self._view)
            if self._handle:
                self._k32.CloseHandle(self._handle)
        finally:
            self._view = self._handle = None


class ACTelemetry:
    """Opens the three shared-memory blocks and reads live snapshots.

    Use as a context manager:

        with ACTelemetry() as tele:
            phys = tele.read_physics()

    Raises FileNotFoundError from open() if AC isn't running.
    """

    def __init__(self) -> None:
        self._physics_block: _MappedBlock | None = None
        self._graphics_block: _MappedBlock | None = None
        self._static_block: _MappedBlock | None = None

    def open(self) -> "ACTelemetry":
        self._physics_block = _MappedBlock(PHYSICS_TAGS, _Physics)
        self._graphics_block = _MappedBlock(GRAPHICS_TAGS, _Graphics)
        self._static_block = _MappedBlock(STATIC_TAGS, _Static)
        return self

    def close(self) -> None:
        for block in (self._physics_block, self._graphics_block, self._static_block):
            if block is not None:
                block.close()

    def __enter__(self) -> "ACTelemetry":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    def read_physics(self) -> PhysicsSnapshot:
        raw = self._physics_block.read()
        return PhysicsSnapshot(
            gas=raw.gas,
            brake=raw.brake,
            steer=raw.steerAngle,
            speed_kmh=raw.speedKmh,
            gear=raw.gear,
            rpm=raw.rpms,
            tyre_core_temp=_quad(raw.tyreCoreTemperature),
            tyre_pressure=_quad(raw.wheelsPressure),
            wheel_slip=_quad(raw.wheelSlip),
            wheel_load=_quad(raw.wheelLoad),
            suspension_travel=_quad(raw.suspensionTravel),
        )

    def read_graphics(self) -> GraphicsSnapshot:
        raw = self._graphics_block.read()
        return GraphicsSnapshot(
            status=STATUS.get(raw.status, str(raw.status)),
            completed_laps=raw.completedLaps,
            current_time_ms=raw.iCurrentTime,
            last_time_ms=raw.iLastTime,
            best_time_ms=raw.iBestTime,
            is_in_pit=bool(raw.isInPit),
            current_sector=raw.currentSectorIndex,
            tyre_compound=raw.tyreCompound,
            car_position=raw.normalizedCarPosition,
        )

    def read_static(self) -> StaticSnapshot:
        raw = self._static_block.read()
        return StaticSnapshot(
            car_model=raw.carModel,
            track=raw.track,
            max_rpm=raw.maxRpm,
        )


def is_available() -> bool:
    """True if the AC shared memory can be opened right now (game process up)."""
    try:
        with ACTelemetry():
            return True
    except (OSError, FileNotFoundError):
        return False


def session_status() -> str:
    """Current game state: OFF (no AC), or LIVE / PAUSE / REPLAY / MENU.

    Distinguishes 'AC process running' from 'driver actually on track' — the app
    should only auto-diagnose from a LIVE (or freshly PAUSEd) driving session.
    """
    try:
        with ACTelemetry() as tele:
            return tele.read_graphics().status
    except (OSError, FileNotFoundError):
        return "OFF"


def read_car_track() -> tuple[str, str]:
    """(car_id, track_id) from the running game. Raises if AC isn't running."""
    with ACTelemetry() as tele:
        s = tele.read_static()
        return s.car_model, s.track
