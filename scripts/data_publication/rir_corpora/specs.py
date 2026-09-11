"""Per-corpus descriptors for rirpub.py: source pins, licences, extraction, card text."""

from __future__ import annotations

import fnmatch
import shutil
import subprocess
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

FETCH = "uv run python -m scripts.data_publication.rir_corpora.fetch_sources"

PRIVATE = "PRIVATE / INTERNAL / NON-COMMERCIAL RESEARCH ONLY"
# macOS archive litter is not part of any release's data and is dropped before inventory.
JUNK_GLOBS = (".DS_Store", "*/.DS_Store", "__MACOSX/*", "*/__MACOSX/*", "Thumbs.db", "*/Thumbs.db")
ANCILLARY_GLOBS = (
    "*.md",
    "*.txt",
    "*.pdf",
    "*.csv",
    "*.json",
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.m",
    "*.py",
    "*.html",
    "*.xml",
    "LICENSE*",
    "README*",
)


def executable(name: str) -> str:
    """Resolve a required external tool to its absolute path.

    :param name: Program name.
    :returns: Absolute path.
    :raises RuntimeError: The program is not on ``PATH``.
    """
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"required tool {name!r} is not on PATH")
    return path


def unzip_all(source: Path, target: Path, names: list[str]) -> None:
    """Extract each archive into a directory named after its stem.

    :param source: Directory holding the archives.
    :param target: Extraction root.
    :param names: Archive file names.
    """
    for name in names:
        with zipfile.ZipFile(source / name) as archive:
            archive.extractall(target / Path(name).stem)


def unzip_split(source: Path, target: Path, zip_name: str) -> None:
    """Reassemble a split ZIP (name.z01..zNN + name.zip) with ``zip -s 0``, then extract.

    :param source: Directory holding the split parts.
    :param target: Extraction root.
    :param zip_name: Name of the final ``.zip`` part.
    """
    joined = target.parent / f"{Path(zip_name).stem}-joined.zip"
    subprocess.run(  # noqa: S603 — args are literal strings
        [executable("zip"), "-q", "-s", "0", str(source / zip_name), "--out", str(joined)],
        check=True,
    )
    with zipfile.ZipFile(joined) as archive:
        archive.extractall(target / Path(zip_name).stem)
    joined.unlink()


def copy_loose(source: Path, target: Path, names: list[str]) -> None:
    """Copy loose (non-archive) source files into the extraction root.

    :param source: Directory holding the files.
    :param target: Extraction root.
    :param names: File names to copy.
    """
    for name in names:
        shutil.copyfile(source / name, target / name)


@dataclass
class Spec:
    """Everything rirpub.py needs to publish one corpus.

    .. attribute :: name

        Corpus name; also the R2 prefix and local directory.

    .. attribute :: title

        Human dataset title used in the card and schema metadata.

    .. attribute :: summary

        One-paragraph description of the corpus.

    .. attribute :: source_urls

        Canonical upstream URLs.

    .. attribute :: pin

        Immutable revision, DOI, or hash statement.

    .. attribute :: license

        Licence summary line.

    .. attribute :: license_note

        Licence detail paragraph for the card.

    .. attribute :: citation

        Required citation.

    .. attribute :: acquisition_commands

        Exact acquisition command lines recorded in the card.

    .. attribute :: archives

        Archives under ``source/`` that ``extract`` unpacks; hash-pinned in the card.

    .. attribute :: loose_files

        Non-archive source files copied into the payload root.

    .. attribute :: payload_root

        Directory (relative to the corpus root) whose files become rows.

    .. attribute :: extractor

        Custom extraction callable, or None for plain ZIP extraction.

    .. attribute :: exclude_globs

        fnmatch patterns relative to the payload root, dropped before inventory.

    .. attribute :: provenance_files

        Files under ``source/`` copied to ``source/_acquisition/``.

    .. attribute :: usage

        Access restriction statement.

    .. attribute :: inclusion

        Inclusion/exclusion statement for the card.

    .. attribute :: split_note

        How consumers should partition the corpus.

    .. attribute :: format_note

        Container-format caveats for the card.

    .. attribute :: caveats

        Known source anomalies.
    """

    name: str
    title: str
    summary: str
    source_urls: list[str]
    pin: str
    license: str
    license_note: str
    citation: str
    acquisition_commands: list[str]
    archives: list[str] = field(default_factory=list)
    loose_files: list[str] = field(default_factory=list)
    payload_root: str = "extracted"
    extractor: Callable[[Path, Path], None] | None = None
    exclude_globs: tuple[str, ...] = ()
    provenance_files: list[str] = field(default_factory=list)
    usage: str = PRIVATE
    inclusion: str = "every object of the pinned release is included; nothing is transcoded. macOS archive litter (`.DS_Store`, `__MACOSX/` resource forks) is dropped before inventory."
    split_note: str = "Consumers partition by room/space according to their protocol."
    format_note: str = ""
    caveats: str = "None observed at build time beyond those stated above."

    def extract(self, source: Path, target: Path) -> None:
        """Unpack archives and copy loose files into the payload root.

        :param source: ``source/`` directory of the corpus.
        :param target: Payload root to populate.
        """
        if self.extractor is not None:
            self.extractor(source, target)
        elif self.archives:
            unzip_all(source, target, self.archives)
        if self.loose_files:
            copy_loose(source, target, self.loose_files)

    def excluded(self, relative: str) -> bool:
        """Return whether a source object is dropped before inventory.

        :param relative: POSIX path relative to the payload root.
        :returns: True for archive litter and corpus-specific exclusions.
        """
        return any(fnmatch.fnmatch(relative, g) for g in (*JUNK_GLOBS, *self.exclude_globs))

    def is_ancillary(self, relative: str) -> bool:
        """Return whether an object is documentation copied as a plain file too.

        :param relative: POSIX path relative to the payload root.
        :returns: True for READMEs, licences, tables, figures, and scripts.
        """
        base = relative.rsplit("/", 1)[-1]
        return any(fnmatch.fnmatch(base, g) for g in ANCILLARY_GLOBS)


def zenodo_commands(name: str, record: int, files: str) -> list[str]:
    """Render the acquisition command lines recorded in a Zenodo corpus card.

    :param name: Corpus name.
    :param record: Zenodo record id.
    :param files: Human summary of the fetched objects.
    :returns: Comment line plus the fetch command.
    """
    return [
        f"# Zenodo record {record}: {files}; md5 verified against the record's file checksums",
        f"{FETCH} zenodo {name}   # https://zenodo.org/api/records/{record}/files/<key>/content",
    ]


SPECS: dict[str, Spec] = {}


def register(spec: Spec) -> None:
    """Add a descriptor to the registry keyed by corpus name.

    :param spec: Corpus descriptor.
    """
    SPECS[spec.name] = spec


register(
    Spec(
        name="MITIRSurvey",
        title="MIT Acoustical Reverberation Scene Statistics Survey (IR Survey)",
        summary="271 mono environmental impulse responses measured at locations sampled from the daily life of seven volunteers (Traer and McDermott, PNAS 2016). Filename prefixes encode the number of location reports (`Nhits`) received for each space.",
        source_urls=[
            "https://mcdermottlab.mit.edu/Reverb/IR_Survey.html",
            "https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip",
        ],
        pin="no upstream version or tag exists; `Audio.zip` is pinned by the SHA-256 recorded in the archive table and acquisition record (fetched 2026-09-08).",
        license="not stated upstream (academic download page); treated as all-rights-reserved research data",
        license_note="The download page states no licence. The corpus is redistributed only inside this private bucket for internal research; contact the McDermott Lab for any other use.",
        citation="J. Traer and J. H. McDermott, “Statistics of natural reverberation enable perceptual separation of sound and space,” Proc. Natl. Acad. Sci. USA, vol. 113, no. 48, pp. E7856–E7865, 2016. DOI: 10.1073/pnas.1612524113.",
        acquisition_commands=[
            f"{FETCH} direct   # https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip"
        ],
        archives=["Audio.zip"],
        caveats="The upstream page advertises 271 IRs, but `Audio.zip` contains 270 WAV objects; no object was dropped by this publication.",
    )
)

register(
    Spec(
        name="AachenIR",
        title="Aachen Impulse Response (AIR) Database, release 1.4",
        summary="Binaural and single-channel room impulse responses measured with a dummy head in offices, lecture rooms, a stairway, corridors, and the Aula Carolina (Jeub, Schäfer, Vary, DSP 2009), distributed as MATLAB MAT-files with loader scripts.",
        source_urls=[
            "https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/",
            "https://www2.iks.rwth-aachen.de/air/air_database_release_1_4.zip",
            "https://www.openslr.org/20/",
        ],
        pin="release 1.4 archive `air_database_release_1_4.zip`, SHA-256 in the archive table (the OpenSLR 20 mirror serves a byte-identical archive).",
        license="not stated in the download (OpenSLR 20: “License: Not stated in the download”); `readme.txt` carries “(c) 2011 RWTH Aachen University”; treated as all-rights-reserved research data",
        license_note="The release ships no licence file; `readme.txt` (preserved under `source/`) states the RWTH Aachen copyright and contact. Contact IKS RWTH Aachen for any use beyond internal research.",
        citation="M. Jeub, M. Schäfer, and P. Vary, “A binaural room impulse response database for the evaluation of dereverberation algorithms,” in Proc. Int. Conf. Digital Signal Processing (DSP), Santorini, Greece, 2009.",
        acquisition_commands=[
            f"{FETCH} direct   # https://www2.iks.rwth-aachen.de/air/air_database_release_1_4.zip"
        ],
        archives=["air_database_release_1_4.zip"],
        format_note="Impulse responses are stored as MAT-files (`application/x-matlab-data`), so `audio_decodable` is false for them; the upstream `load_air.m` loader documents the structure.",
    )
)

register(
    Spec(
        name="EchoThief",
        title="EchoThief Impulse Response Library",
        summary="Over one hundred impulse responses of unusual real-world spaces across North America, recorded by Chris Warren (San Diego State University) and distributed as WAV files.",
        source_urls=[
            "https://www.echothief.com/",
            "https://www.echothief.com/downloads/",
            "https://www.echothief.com/wp-content/uploads/2024/07/EchoThiefImpulseResponseLibrary.zip",
        ],
        pin="no upstream version exists; the July-2024 archive is pinned by the SHA-256 recorded in the archive table and acquisition record (fetched 2026-09-08).",
        license="custom EchoThief licence: derivative works such as convolution reverberation are permitted; any other use requires the author's permission",
        license_note="Upstream licence text: “You are welcome to use the EchoThief Impulse Response Library to create derivative work (such as convolving it with other sounds to create reverberation). If you would like to use it in any other way, let's talk.” Copyright 2013–2026 Dr. Chris Warren.",
        citation="C. Warren, “EchoThief impulse response library,” http://www.echothief.com/ (accessed 2026-09-08).",
        acquisition_commands=[
            f"{FETCH} direct   # https://www.echothief.com/wp-content/uploads/2024/07/EchoThiefImpulseResponseLibrary.zip"
        ],
        archives=["EchoThiefImpulseResponseLibrary.zip"],
    )
)

register(
    Spec(
        name="ASHIR",
        title="ASH-IR Dataset (Audio Spatialisation for Headphones Impulse Responses)",
        summary="Binaural room impulse responses (2-channel WAV, 44.1 kHz) derived from public HATS measurements in many rooms and equalised for diffuse-field headphones, plus headphone compensation filters and Equalizer APO configuration files (Shanon Pearce).",
        source_urls=["https://github.com/ShanonPearce/ASH-IR-Dataset"],
        pin="Git revision `af0d3f51e19cf74dc77a603fd3135a14d8aeb29e` (branch `master`, 2024-06-14); the upstream repository publishes no tags.",
        license="CC BY-NC-SA 4.0 (upstream `LICENSE.txt`)",
        license_note="Upstream README: “Unless otherwise stated, all files in this repository are licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0.” The BRIRs are themselves derived from third-party HATS datasets credited in the upstream wiki.",
        citation="S. Pearce, “ASH-IR dataset,” https://github.com/ShanonPearce/ASH-IR-Dataset, revision af0d3f5 (2024-06-14).",
        acquisition_commands=[
            f"{FETCH} ashir   # git clone --mirror + worktree add --detach af0d3f51e19cf74dc77a603fd3135a14d8aeb29e",
        ],
        payload_root="source",
        exclude_globs=(".git", ".git/*"),
        split_note="Consumers partition by room according to their protocol.",
    )
)

register(
    Spec(
        name="MultiRoomTransition",
        title="Multi-Room Transition Dataset (MRTD)",
        summary="Room impulse responses measured along transitions through coupled multi-room environments (offices, workshops, hallway–lecture-hall) with a KEMAR dummy head and a Zoom recorder for four loudspeaker positions, plus a pickled dataframe of derived metadata (Götz et al., IWAENC 2024).",
        source_urls=[
            "https://doi.org/10.5281/zenodo.13341566",
            "https://zenodo.org/records/13341566",
            "https://github.com/audiolabs/blind-multi-room-model",
        ],
        pin="Zenodo record 13341566 (DOI 10.5281/zenodo.13341566, concept DOI 10.5281/zenodo.11196819); every file md5-verified against the record and SHA-256 inventoried.",
        license="CC BY 4.0",
        license_note="Zenodo licence field: Creative Commons Attribution 4.0 International.",
        citation="P. Götz, G. Götz, S. J. Schlecht, V. Pulkki, and E. A. P. Habets, “A multi-room transition dataset for blind estimation of energy decay,” in Proc. IWAENC, Sep. 2024, pp. 125–129. Dataset DOI: 10.5281/zenodo.13341566.",
        acquisition_commands=zenodo_commands(
            "MultiRoomTransition", 13341566, "25 SOFA files + mrtd_dataframe.pkl"
        ),
        payload_root="source",
        exclude_globs=("zenodo-record-*.json", "acquisition.json"),
        provenance_files=["zenodo-record-13341566.json"],
        format_note="SOFA (HDF5) containers and a Python pickle are stored byte-exact; `audio_decodable` is false for them. Reading requires a SOFA/HDF5 reader and, for the pickle, the upstream pandas environment.",
        caveats="The pickle is only loadable with a compatible pandas version; it is preserved as an opaque object.",
    )
)

register(
    Spec(
        name="THKoelnSRIR",
        title="A High-Resolution Spatial Room Impulse Response Database (TH Köln)",
        summary="Spatial RIRs of three rooms (Audiolab, Classroom, Audimax) for three source and two receiver positions each: Lebedev-2702 spherical-array DRIRs and KU100 360° BRIRs as SOFA files, plus omnidirectional WAV impulse responses and floor plans (Lübeck, Arend, Pörschmann, DAGA 2021).",
        source_urls=[
            "https://doi.org/10.5281/zenodo.5031335",
            "https://zenodo.org/records/5031335",
        ],
        pin="Zenodo record 5031335 (DOI 10.5281/zenodo.5031335, concept DOI 10.5281/zenodo.5031334); every file md5-verified against the record and SHA-256 inventoried.",
        license="CC BY 4.0",
        license_note="Zenodo licence field: Creative Commons Attribution 4.0 International.",
        citation="T. Lübeck, J. M. Arend, and C. Pörschmann, “A high-resolution spatial room impulse response database,” in Fortschritte der Akustik – DAGA 2021, Vienna, 2021. Dataset DOI: 10.5281/zenodo.5031335.",
        acquisition_commands=zenodo_commands(
            "THKoelnSRIR", 5031335, "58 objects: SOFA DRIR/BRIR files, omni WAVs, floor plans"
        ),
        payload_root="source",
        exclude_globs=("zenodo-record-*.json", "acquisition.json"),
        provenance_files=["zenodo-record-5031335.json"],
        format_note="SOFA (HDF5) containers are stored byte-exact with `audio_decodable` false; the omnidirectional `Omni_ir_*.wav` objects decode natively.",
    )
)

register(
    Spec(
        name="MPRIR",
        title="Multi-Purpose Room Impulse Response Dataset Measured on a 3D Spatial Grid (MP-RIR)",
        summary="68,736 RIRs measured by a robot on a dense 3D grid of 8,592 microphone positions for eight source configurations in a complex-shaped room at Fraunhofer IIS (Friede et al., AES 156th Convention, 2024), distributed as one NumPy array per source plus grid coordinates and setup metadata.",
        source_urls=[
            "https://doi.org/10.5281/zenodo.11148712",
            "https://zenodo.org/records/11148712",
            "https://www.dsai.iis.fraunhofer.com/extensive-room-impulse-response-dataset-published/",
        ],
        pin="Zenodo record 11148712 (DOI 10.5281/zenodo.11148712, concept DOI 10.5281/zenodo.11148711); every file md5-verified against the record and SHA-256 inventoried.",
        license="CC BY 4.0",
        license_note="Zenodo licence field: Creative Commons Attribution 4.0 International.",
        citation="L. Friede, C. Heuberger, K. Prawda, S. J. Schlecht, E. A. P. Habets, et al., “Multi-purpose room impulse response dataset measured on a 3D spatial grid,” in Proc. 156th AES Convention, Madrid, Jun. 2024. Dataset DOI: 10.5281/zenodo.11148712.",
        acquisition_commands=zenodo_commands(
            "MPRIR", 11148712, "S1..S8_Mrir.npy, Mxyz.npy, Setup.npz"
        ),
        payload_root="source",
        exclude_globs=("zenodo-record-*.json", "acquisition.json"),
        provenance_files=["zenodo-record-11148712.json"],
        format_note="Each `S<n>_Mrir.npy` is a single 3.44 GB NumPy array stored as one blob row; `audio_decodable` is false for every object. Per-RIR unpacking is a derived dataset, not part of this publication.",
    )
)

register(
    Spec(
        name="TAUSRIR",
        title="TAU Spatial Room Impulse Response Database (TAU-SRIR DB)",
        summary="Spatial RIRs captured with an Eigenmike in nine rooms of Tampere University along circular and linear source trajectories, extracted at roughly 1° spacing in 4-channel MIC and FOA formats at 24 kHz, stored per room in MAT-files with DOA and measurement metadata (Politis, Adavanne, Virtanen, 2022).",
        source_urls=[
            "https://doi.org/10.5281/zenodo.6408611",
            "https://zenodo.org/records/6408611",
        ],
        pin="Zenodo record 6408611 (DOI 10.5281/zenodo.6408611, concept DOI 10.5281/zenodo.6408610); split ZIP parts md5-verified against the record and SHA-256 inventoried.",
        license="custom open non-commercial licence with attribution (upstream `LICENSE.txt`)",
        license_note="Upstream README: “The database is published under a custom open non-commercial with attribution license. It can be found in the LICENSE.txt file that accompanies the data.” The licence text is preserved under `source/`.",
        citation="A. Politis, S. Adavanne, and T. Virtanen, “TAU spatial room impulse response database (TAU-SRIR DB),” Zenodo, Apr. 2022, DOI: 10.5281/zenodo.6408611. See also A. Politis, S. Adavanne, T. Virtanen, “A dataset of reverberant spatial sound scenes with moving sources for sound event localization and detection,” Proc. DCASE 2020 Workshop.",
        acquisition_commands=zenodo_commands(
            "TAUSRIR",
            6408611,
            "TAU-SRIR_DB.zip + .z01-.z03, README.md, LICENSE.txt (TAU-SNoise_DB parts intentionally not fetched)",
        ),
        archives=["TAU-SRIR_DB.zip", "TAU-SRIR_DB.z01", "TAU-SRIR_DB.z02", "TAU-SRIR_DB.z03"],
        loose_files=["README.md", "LICENSE.txt"],
        extractor=lambda source, target: unzip_split(source, target, "TAU-SRIR_DB.zip"),
        provenance_files=["zenodo-record-6408611.json"],
        inclusion="the SRIR archive and its README/LICENSE are included in full; the separate spatial ambient-noise archive (`TAU-SNoise_DB`) is excluded because only impulse responses are needed.",
        format_note="RIRs are MAT-files (`application/x-matlab-data`), so `audio_decodable` is false; `rirdata.mat` and `measinfo.mat` carry DOAs and room metadata.",
    )
)

register(
    Spec(
        name="Arni",
        title="Dataset of impulse responses from variable acoustics room Arni at Aalto Acoustic Labs",
        summary="Impulse responses of 5,342 absorption-panel configurations of the variable-acoustics laboratory Arni (Aalto University), measured with an omnidirectional source and several microphones; each configuration's panel state is listed in `combinations_setup.csv` (Prawda, Schlecht, Välimäki, 2022).",
        source_urls=[
            "https://doi.org/10.5281/zenodo.6985104",
            "https://zenodo.org/records/6985104",
        ],
        pin="Zenodo record 6985104 (DOI 10.5281/zenodo.6985104, concept DOI 10.5281/zenodo.6985103); six ZIP archives md5-verified against the record and SHA-256 inventoried.",
        license="CC BY 4.0",
        license_note="Zenodo licence field: Creative Commons Attribution 4.0 International.",
        citation="K. Prawda, S. J. Schlecht, and V. Välimäki, “Dataset of impulse responses from variable acoustics room Arni at Aalto Acoustic Labs,” Zenodo, 2022, DOI: 10.5281/zenodo.6985104.",
        acquisition_commands=zenodo_commands(
            "Arni",
            6985104,
            "six IR_Arni_upload_numClosed_*.zip, combinations_setup.csv, Arni_layout.jpg, Arni_panels_numbers.pdf",
        ),
        archives=[
            "IR_Arni_upload_numClosed_0-5.zip",
            "IR_Arni_upload_numClosed_6-15.zip",
            "IR_Arni_upload_numClosed_16-25.zip",
            "IR_Arni_upload_numClosed_26-35.zip",
            "IR_Arni_upload_numClosed_36-45.zip",
            "IR_Arni_upload_numClosed_46-55.zip",
        ],
        loose_files=["combinations_setup.csv", "Arni_layout.jpg", "Arni_panels_numbers.pdf"],
        provenance_files=["zenodo-record-6985104.json"],
        split_note="Consumers partition by panel configuration according to their protocol.",
    )
)

register(
    Spec(
        name="OpenAIR",
        title="OpenAIR – Open Acoustic Impulse Response Library (University of York file store mirror)",
        summary="Impulse responses of buildings, heritage sites, and other spaces contributed to the University of York OpenAIR library, in B-format, stereo, and mono renderings with per-entry documentation, images, and example auralisations (Murphy and Shelley, AES 128th Convention, 2010).",
        source_urls=[
            "https://webfiles.york.ac.uk/OPENAIR/",
            "https://openairlib.net/",
            "https://www.york.ac.uk/physics-engineering-technology/research/communication-technologies/projects/open-acoustic-impulse-response-library/",
        ],
        pin="no upstream version exists and `openairlib.net` was suspended at fetch time; the York file store was crawled recursively on 2026-09-08 and every object is pinned by SHA-256 in the inventory.",
        license="per-entry Creative Commons licences chosen by contributors (mostly CC BY 4.0); see each entry's documentation under `source/`",
        license_note="OpenAIR states that “most of the content on this site is released under Creative Commons licenses” chosen per contribution; the per-entry licence documents shipped in the file store are preserved. Entries without an explicit licence are treated as non-commercial research only.",
        citation="D. T. Murphy and S. Shelley, “OpenAIR: An interactive auralization web resource and database,” in Proc. 129th AES Convention, San Francisco, Nov. 2010.",
        acquisition_commands=[
            f"{FETCH} openair   # wget -m -np -nH --cut-dirs=1 https://webfiles.york.ac.uk/OPENAIR/"
        ],
        payload_root="source",
        exclude_globs=("*index.html*", "*/index.html*", "*?C=*", ".index/*", "*/.index/*"),
        inclusion="every object served by the York file store crawl (`IRs/`, `Anechoic/`, `Resources/`, `Turntable/`) is included, except the server-generated directory index pages; per-entry ZIP bundles are kept alongside their unpacked audio because both are served upstream.",
        split_note="Consumers partition by entry (space) according to their protocol.",
        caveats="The mirror reflects the York file store on the crawl date; `openairlib.net` metadata pages were unavailable (account suspended), so only the documentation shipped inside the file store is preserved.",
    )
)

register(
    Spec(
        name="ACE",
        title="ACE Challenge corpus (room impulse responses)",
        summary="Blocked: acquisition requires ACE corpus registration; see issue #3177.",
        source_urls=[
            "http://www.ee.ic.ac.uk/naylor/ACEweb/index.html",
            "https://ieee-dataport.org/documents/ace-challenge-2015",
        ],
        pin="not acquired",
        license="CC BY-ND 4.0",
        license_note="",
        citation="J. Eaton, N. D. Gaubitch, A. H. Moore, and P. A. Naylor, “Estimation of room acoustic parameters: The ACE challenge,” IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 24, no. 10, pp. 1681–1693, 2016.",
        acquisition_commands=[],
    )
)
