import importlib
import importlib.util
import inspect
import json
import os
import platform
import signal
import subprocess
import sys
import time
import urllib.request

import pytest

import matplotlib as mpl
from matplotlib import _c_internal_utils


# Minimal smoke-testing of the backends for which the dependencies are
# PyPI-installable on CI.  They are not available for all tested Python
# versions so we don't fail on missing backends.

def _get_testable_interactive_backends():
    envs = []
    for deps, env in [
            *[([qt_api],
               {"MPLBACKEND": "qtagg", "QT_API": qt_api})
              for qt_api in ["PyQt6", "PySide6", "PyQt5", "PySide2"]],
            *[([qt_api, "cairocffi"],
               {"MPLBACKEND": "qtcairo", "QT_API": qt_api})
              for qt_api in ["PyQt6", "PySide6", "PyQt5", "PySide2"]],
            *[(["cairo", "gi"], {"MPLBACKEND": f"gtk{version}{renderer}"})
              for version in [3, 4] for renderer in ["agg", "cairo"]],
            (["tkinter"], {"MPLBACKEND": "tkagg"}),
            (["wx"], {"MPLBACKEND": "wx"}),
            (["wx"], {"MPLBACKEND": "wxagg"}),
            (["matplotlib.backends._macosx"], {"MPLBACKEND": "macosx"}),
    ]:
        reason = None
        missing = [dep for dep in deps if not importlib.util.find_spec(dep)]
        if (sys.platform == "linux" and
                not _c_internal_utils.display_is_valid()):
            reason = "$DISPLAY and $WAYLAND_DISPLAY are unset"
        elif missing:
            reason = "{} cannot be imported".format(", ".join(missing))
        elif env["MPLBACKEND"] == 'macosx' and os.environ.get('TF_BUILD'):
            reason = "macosx backend fails on Azure"
        elif env["MPLBACKEND"].startswith('gtk'):
            import gi
            version = env["MPLBACKEND"][3]
            repo = gi.Repository.get_default()
            if f'{version}.0' not in repo.enumerate_versions('Gtk'):
                reason = "no usable GTK bindings"
        marks = []
        if reason:
            marks.append(pytest.mark.skip(
                reason=f"Skipping {env} because {reason}"))
        elif env["MPLBACKEND"].startswith('wx') and sys.platform == 'darwin':
            # ignore on OSX because that's currently broken (github #16849)
            marks.append(pytest.mark.xfail(reason='github #16849'))
        envs.append(pytest.param(env, marks=marks, id=str(env)))
    return envs


_test_timeout = 60  # A reasonably safe value for slower architectures.


# The source of this function gets extracted and run in another process, so it
# must be fully self-contained.
# Using a timer not only allows testing of timers (on other backends), but is
# also necessary on gtk3 and wx, where a direct call to key_press_event("q")
# from draw_event causes breakage due to the canvas widget being deleted too
# early.  Also, gtk3 redefines key_press_event with a different signature, so
# we directly invoke it from the superclass instead.
def _test_interactive_impl():
    import importlib.util
    import io
    import json
    import sys
    from unittest import TestCase

    import matplotlib as mpl
    from matplotlib import pyplot as plt, rcParams
    from matplotlib.backend_bases import FigureCanvasBase

    rcParams.update({
        "webagg.open_in_browser": False,
        "webagg.port_retries": 1,
    })
    if len(sys.argv) >= 2:  # Second argument is json-encoded rcParams.
        rcParams.update(json.loads(sys.argv[1]))
    backend = plt.rcParams["backend"].lower()
    assert_equal = TestCase().assertEqual
    assert_raises = TestCase().assertRaises

    if backend.endswith("agg") and not backend.startswith(("gtk", "web")):
        # Force interactive framework setup.
        plt.figure()

        # Check that we cannot switch to a backend using another interactive
        # framework, but can switch to a backend using cairo instead of agg,
        # or a non-interactive backend.  In the first case, we use tkagg as
        # the "other" interactive backend as it is (essentially) guaranteed
        # to be present.  Moreover, don't test switching away from gtk3 (as
        # Gtk.main_level() is not set up at this point yet) and webagg (which
        # uses no interactive framework).

        if backend != "tkagg":
            with assert_raises(ImportError):
                mpl.use("tkagg", force=True)

        def check_alt_backend(alt_backend):
            mpl.use(alt_backend, force=True)
            fig = plt.figure()
            assert_equal(
                type(fig.canvas).__module__,
                "matplotlib.backends.backend_{}".format(alt_backend))

        if importlib.util.find_spec("cairocffi"):
            check_alt_backend(backend[:-3] + "cairo")
        check_alt_backend("svg")

    mpl.use(backend, force=True)

    fig, ax = plt.subplots()
    assert_equal(
        type(fig.canvas).__module__,
        "matplotlib.backends.backend_{}".format(backend))

    ax.plot([0, 1], [2, 3])
    if fig.canvas.toolbar:  # i.e toolbar2.
        fig.canvas.toolbar.draw_rubberband(None, 1., 1, 2., 2)

    timer = fig.canvas.new_timer(1.)  # Test floats casting to int as needed.
    timer.add_callback(FigureCanvasBase.key_press_event, fig.canvas, "q")
    # Trigger quitting upon draw.
    fig.canvas.mpl_connect("draw_event", lambda event: timer.start())
    fig.canvas.mpl_connect("close_event", print)

    result = io.BytesIO()
    fig.savefig(result, format='png')

    plt.show()

    # Ensure that the window is really closed.
    plt.pause(0.5)

    # Test that saving works after interactive window is closed, but the figure
    # is not deleted.
    result_after = io.BytesIO()
    fig.savefig(result_after, format='png')

    if not backend.startswith('qt5') and sys.platform == 'darwin':
        # FIXME: This should be enabled everywhere once Qt5 is fixed on macOS
        # to not resize incorrectly.
        assert_equal(result.getvalue(), result_after.getvalue())


@pytest.mark.parametrize("env", _get_testable_interactive_backends())
@pytest.mark.parametrize("toolbar", ["toolbar2", "toolmanager"])
@pytest.mark.flaky(reruns=3)
def test_interactive_backend(env, toolbar):
    if env["MPLBACKEND"] == "macosx":
        if toolbar == "toolmanager":
            pytest.skip("toolmanager is not implemented for macosx.")

    proc = subprocess.run(
        [sys.executable, "-c",
         inspect.getsource(_test_interactive_impl)
         + "\n_test_interactive_impl()",
         json.dumps({"toolbar": toolbar})],
        env={**os.environ, "SOURCE_DATE_EPOCH": "0", **env},
        timeout=_test_timeout,
        stdout=subprocess.PIPE, universal_newlines=True)
    if proc.returncode:
        pytest.fail("The subprocess returned with non-zero exit status "
                    f"{proc.returncode}.")
    assert proc.stdout.count("CloseEvent") == 1


# The source of this function gets extracted and run in another process, so it
# must be fully self-contained.
def _test_thread_impl():
    from concurrent.futures import ThreadPoolExecutor
    import json
    import sys

    from matplotlib import pyplot as plt, rcParams

    rcParams.update({
        "webagg.open_in_browser": False,
        "webagg.port_retries": 1,
    })
    if len(sys.argv) >= 2:  # Second argument is json-encoded rcParams.
        rcParams.update(json.loads(sys.argv[1]))

    # Test artist creation and drawing does not crash from thread
    # No other guarantees!
    fig, ax = plt.subplots()
    # plt.pause needed vs plt.show(block=False) at least on toolbar2-tkagg
    plt.pause(0.5)

    future = ThreadPoolExecutor().submit(ax.plot, [1, 3, 6])
    future.result()  # Joins the thread; rethrows any exception.

    fig.canvas.mpl_connect("close_event", print)
    future = ThreadPoolExecutor().submit(fig.canvas.draw)
    plt.pause(0.5)  # flush_events fails here on at least Tkagg (bpo-41176)
    future.result()  # Joins the thread; rethrows any exception.
    plt.close()
    fig.canvas.flush_events()  # pause doesn't process events after close


_thread_safe_backends = _get_testable_interactive_backends()
# Known unsafe backends. Remove the xfails if they start to pass!
for param in _thread_safe_backends:
    backend = param.values[0]["MPLBACKEND"]
    if "cairo" in backend:
        # Cairo backends save a cairo_t on the graphics context, and sharing
        # these is not threadsafe.
        param.marks.append(
            pytest.mark.xfail(raises=subprocess.CalledProcessError))
    elif backend == "wx":
        param.marks.append(
            pytest.mark.xfail(raises=subprocess.CalledProcessError))
    elif backend == "macosx":
        from packaging.version import parse
        mac_ver = platform.mac_ver()[0]
        # Note, macOS Big Sur is both 11 and 10.16, depending on SDK that
        # Python was compiled against.
        if mac_ver and parse(mac_ver) < parse('10.16'):
            param.marks.append(
                pytest.mark.xfail(raises=subprocess.TimeoutExpired,
                                  strict=True))
    elif param.values[0].get("QT_API") == "PySide2":
        param.marks.append(
            pytest.mark.xfail(raises=subprocess.CalledProcessError))
    elif backend == "tkagg" and platform.python_implementation() != 'CPython':
        param.marks.append(
            pytest.mark.xfail(
                reason='PyPy does not support Tkinter threading: '
                       'https://foss.heptapod.net/pypy/pypy/-/issues/1929',
                strict=True))


@pytest.mark.parametrize("env", _thread_safe_backends)
@pytest.mark.flaky(reruns=3)
def test_interactive_thread_safety(env):
    proc = subprocess.run(
        [sys.executable, "-c",
         inspect.getsource(_test_thread_impl) + "\n_test_thread_impl()"],
        env={**os.environ, "SOURCE_DATE_EPOCH": "0", **env},
        timeout=_test_timeout, check=True,
        stdout=subprocess.PIPE, universal_newlines=True)
    assert proc.stdout.count("CloseEvent") == 1


@pytest.mark.skipif('TF_BUILD' in os.environ,
                    reason="this test fails an azure for unknown reasons")
@pytest.mark.skipif(os.name == "nt", reason="Cannot send SIGINT on Windows.")
def test_webagg():
    pytest.importorskip("tornado")
    proc = subprocess.Popen(
        [sys.executable, "-c",
         inspect.getsource(_test_interactive_impl)
         + "\n_test_interactive_impl()"],
        env={**os.environ, "MPLBACKEND": "webagg", "SOURCE_DATE_EPOCH": "0"})
    url = "http://{}:{}".format(
        mpl.rcParams["webagg.address"], mpl.rcParams["webagg.port"])
    timeout = time.perf_counter() + _test_timeout
    while True:
        try:
            retcode = proc.poll()
            # check that the subprocess for the server is not dead
            assert retcode is None
            conn = urllib.request.urlopen(url)
            break
        except urllib.error.URLError:
            if time.perf_counter() > timeout:
                pytest.fail("Failed to connect to the webagg server.")
            else:
                continue
    conn.close()
    proc.send_signal(signal.SIGINT)
    assert proc.wait(timeout=_test_timeout) == 0


@pytest.mark.skipif(sys.platform != "linux", reason="this a linux-only test")
@pytest.mark.backend('QtAgg', skip_on_importerror=True)
def test_lazy_linux_headless():
    test_script = """
import os
import sys

# make it look headless
os.environ.pop('DISPLAY', None)
os.environ.pop('WAYLAND_DISPLAY', None)

# we should fast-track to Agg
import matplotlib.pyplot as plt
plt.get_backend() == 'agg'
assert 'PyQt5' not in sys.modules

# make sure we really have pyqt installed
import PyQt5
assert 'PyQt5' in sys.modules

# try to switch and make sure we fail with ImportError
try:
    plt.switch_backend('qt5agg')
except ImportError:
    ...
else:
    sys.exit(1)

"""
    proc = subprocess.run([sys.executable, "-c", test_script],
                          env={**os.environ, "MPLBACKEND": ""})
    if proc.returncode:
        pytest.fail("The subprocess returned with non-zero exit status "
                    f"{proc.returncode}.")


def _test_number_of_draws_script():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    # animated=True tells matplotlib to only draw the artist when we
    # explicitly request it
    ln, = ax.plot([0, 1], [1, 2], animated=True)

    # make sure the window is raised, but the script keeps going
    plt.show(block=False)
    plt.pause(0.1)
    # Connect to draw_event to count the occurrences
    fig.canvas.mpl_connect('draw_event', print)

    # get copy of entire figure (everything inside fig.bbox) sans animated artist
    bg = fig.canvas.copy_from_bbox(fig.bbox)
    # draw the animated artist, this uses a cached renderer
    ax.draw_artist(ln)
    # show the result to the screen, this pushes the updated RGBA buffer from the
    # renderer to the GUI framework so you can see it
    fig.canvas.blit(fig.bbox)

    for j in range(20):
        # reset the background back in the canvas state, screen unchanged
        fig.canvas.restore_region(bg)
        # Create a **new** artist here, this is poor usage of blitting
        # but good for testing to make sure that this doesn't create
        # excessive draws
        ln, = ax.plot([0, 1], [1, 2])
        # render the artist, updating the canvas state, but not the screen
        ax.draw_artist(ln)
        # copy the image to the GUI state, but screen might not changed yet
        fig.canvas.blit(fig.bbox)
        # flush any pending GUI events, re-painting the screen if needed
        fig.canvas.flush_events()

    # Let the event loop process everything before leaving
    plt.pause(0.1)


_blit_backends = _get_testable_interactive_backends()
for param in _blit_backends:
    backend = param.values[0]["MPLBACKEND"]
    if backend == "gtk3cairo":
        # copy_from_bbox only works when rendering to an ImageSurface
        # xfail doesn't work here due to other python errors
        param.marks.append(
            pytest.mark.skip(
                "gtk3cairo does not support blitting"))
    elif backend == "wx":
        param.marks.append(
            pytest.mark.skip("wx does not support blitting"))


@pytest.mark.parametrize("env", _blit_backends)
# subprocesses can struggle to get the display, so rerun
# a few times
@pytest.mark.flaky(reruns=4)
def test_blitting_events(env):
    # FIXME: It would be nice to test all of these backends
    # if env["MPLBACKEND"].startswith("gtk"):
    #     pytest.skip("GTK animation calls draw too many times")
    # if env["MPLBACKEND"].startswith("wx"):
    #     pytest.skip("wx animation calls draw too many times")
    # if env["MPLBACKEND"] == "gtk3cairo":
    #     pytest.skip("gtk3cairo can only use copy_from_bbox on an ImageSurface")
    # if env["MPLBACKEND"] == "qtcairo":
    #     pytest.skip("qtcairo can not get the display on the CI systems")
    # if env.get("QT_API") == "PySide2":
    #     pytest.skip("pyside2 can't open display on CI")

    proc = subprocess.run(
        [sys.executable, "-c",
         inspect.getsource(_test_number_of_draws_script)
         + "\n_test_number_of_draws_script()"],
        env={**os.environ, "SOURCE_DATE_EPOCH": "0", **env},
        timeout=_test_timeout,
        stdout=subprocess.PIPE, universal_newlines=True)

    # Count the number of draw_events we got. We will count some initial
    # canvas draws (which vary in number by backend), but the critical
    # check here is that it isn't 20 draws, which would be called if
    # blitting is not properly implemented
    ndraws = proc.stdout.count("DrawEvent")
    assert 0 < ndraws < 5
