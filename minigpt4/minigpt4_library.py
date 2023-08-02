import os
import sys
import ctypes
import pathlib
from typing import Optional, List
import enum
from pathlib import Path

class DataType(enum.IntEnum):
    def __str__(self):
        return str(self.name)
    
    F16 = 0
    F32 = 1
    I32 = 2
    L64 = 3
    Q4_0 = 4
    Q4_1 = 5
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15

class Verbosity(enum.IntEnum):
    SILENT = 0
    ERR = 1
    INFO = 2
    DEBUG = 3

class ImageFormat(enum.IntEnum):
    UNKNOWN = 0
    F32 = 1
    U8 = 2

I32 = ctypes.c_int32
U32 = ctypes.c_uint32
F32 = ctypes.c_float
SIZE_T = ctypes.c_size_t
VOID_PTR = ctypes.c_void_p
CHAR_PTR = ctypes.POINTER(ctypes.c_char)
FLOAT_PTR = ctypes.POINTER(ctypes.c_float)
INT_PTR = ctypes.POINTER(ctypes.c_int32)
CHAR_PTR_PTR = ctypes.POINTER(ctypes.POINTER(ctypes.c_char))

MiniGPT4ContextP = VOID_PTR
class MiniGPT4Context:
    def __init__(self, ptr: ctypes.pointer):
        self.ptr = ptr

class MiniGPT4Image(ctypes.Structure):
    _fields_ = [
        ('data', VOID_PTR),
        ('width', I32),
        ('height', I32),
        ('channels', I32),
        ('format', I32)
    ]

class MiniGPT4Embedding(ctypes.Structure):
    _fields_ = [
        ('data', FLOAT_PTR),
        ('n_embeddings', SIZE_T),
    ]

MiniGPT4ImageP = ctypes.POINTER(MiniGPT4Image)
MiniGPT4EmbeddingP = ctypes.POINTER(MiniGPT4Embedding)

class MiniGPT4SharedLibrary:
    """
    Python wrapper around minigpt4.cpp shared library.
    """

    def __init__(self, shared_library_path: str):
        """
        Loads the shared library from specified file.
        In case of any error, this method will throw an exception.

        Parameters
        ----------
        shared_library_path : str
            Path to minigpt4.cpp shared library. On Windows, it would look like 'minigpt4.dll'. On UNIX, 'minigpt4.so'.
        """

        self.library = ctypes.cdll.LoadLibrary(shared_library_path)
        if self.library is None:
            raise RuntimeError(f'Failed to load shared library from {shared_library_path}')

        self.library.minigpt4_model_load.argtypes = [
            CHAR_PTR, # const char *path
            CHAR_PTR, # const char *llm_model
            I32, # int verbosity
            I32, # int seed
            I32, # int n_ctx
            I32, # int n_batch
            I32, # int numa
        ]
        self.library.minigpt4_model_load.restype = MiniGPT4ContextP

        self.library.minigpt4_image_load_from_file.argtypes = [
            MiniGPT4ContextP, # struct MiniGPT4Context *ctx
            CHAR_PTR, # const char *path
            MiniGPT4ImageP, # struct MiniGPT4Image *image
            I32, # int flags
        ]
        self.library.minigpt4_image_load_from_file.restype = I32

        self.library.minigpt4_encode_image.argtypes = [
            MiniGPT4ContextP, # struct MiniGPT4Context *ctx
            MiniGPT4ImageP, # const struct MiniGPT4Image *image
            MiniGPT4EmbeddingP, # struct MiniGPT4Embedding *embedding
            I32, # size_t n_threads
        ]
        self.library.minigpt4_encode_image.restype = I32

        self.library.minigpt4_begin_chat_image.argtypes = [
            MiniGPT4ContextP, # struct MiniGPT4Context *ctx
            MiniGPT4EmbeddingP, # struct MiniGPT4Embedding *embedding
            CHAR_PTR, # const char *s
            I32, # size_t n_threads
        ]
        self.library.minigpt4_begin_chat_image.restype = I32

        self.library.minigpt4_end_chat_image.argtypes = [
            MiniGPT4ContextP, # struct MiniGPT4Context *ctx
            CHAR_PTR_PTR, # const char **token
            I32, # size_t n_threads
            F32, # float temp
            I32, # int32_t top_k
            F32, # float top_p
            F32, # float tfs_z
            F32, # float typical_p
            I32, # int32_t repeat_last_n
            F32, # float repeat_penalty
            F32, # float alpha_presence
            F32, # float alpha_frequency
            I32, # int mirostat
            F32, # float mirostat_tau
            F32, # float mirostat_eta
            I32, # int penalize_nl
        ]
        self.library.minigpt4_end_chat_image.restype = I32

        self.library.minigpt4_system_prompt.argtypes = [
            MiniGPT4ContextP, # struct MiniGPT4Context *ctx
            I32, # size_t n_threads
        ]
        self.library.minigpt4_system_prompt.restype = I32

        self.library.minigpt4_begin_chat.argtypes = [
            MiniGPT4ContextP, # struct MiniGPT4Context *ctx
            CHAR_PTR, # const char *s
            I32, # size_t n_threads
        ]
        self.library.minigpt4_begin_chat.restype = I32

        self.library.minigpt4_end_chat.argtypes = [
            MiniGPT4ContextP, # struct MiniGPT4Context *ctx
            CHAR_PTR_PTR, # const char **token
            I32, # size_t n_threads
            F32, # float temp
            I32, # int32_t top_k
            F32, # float top_p
            F32, # float tfs_z
            F32, # float typical_p
            I32, # int32_t repeat_last_n
            F32, # float repeat_penalty
            F32, # float alpha_presence
            F32, # float alpha_frequency
            I32, # int mirostat
            F32, # float mirostat_tau
            F32, # float mirostat_eta
            I32, # int penalize_nl
        ]
        self.library.minigpt4_end_chat.restype = I32

        self.library.minigpt4_reset_chat.argtypes = [
            MiniGPT4ContextP, # struct MiniGPT4Context *ctx
        ]
        self.library.minigpt4_reset_chat.restype = I32

        self.library.minigpt4_contains_eos_token.argtypes = [
            CHAR_PTR, # const char *s
        ]
        self.library.minigpt4_contains_eos_token.restype = I32

        self.library.minigpt4_is_eos.argtypes = [
            CHAR_PTR, # const char *s
        ]
        self.library.minigpt4_is_eos.restype = I32

        self.library.minigpt4_free.argtypes = [
            MiniGPT4ContextP, # struct MiniGPT4Context *ctx
        ]
        self.library.minigpt4_free.restype = I32

        self.library.minigpt4_free_image.argtypes = [
            MiniGPT4ImageP, # struct MiniGPT4Image *image
        ]
        self.library.minigpt4_free_image.restype = I32

        self.library.minigpt4_free_embedding.argtypes = [
            MiniGPT4EmbeddingP, # struct MiniGPT4Embedding *embedding
        ]
        self.library.minigpt4_free_embedding.restype = I32

        self.library.minigpt4_error_code_to_string.argtypes = [
            I32, # int error_code
        ]
        self.library.minigpt4_error_code_to_string.restype = CHAR_PTR

        self.library.minigpt4_quantize_model.argtypes = [
            CHAR_PTR, # const char *in_path
            CHAR_PTR, # const char *out_path
            I32, # int data_type
        ]
        self.library.minigpt4_quantize_model.restype = I32

        self.library.minigpt4_set_verbosity.argtypes = [
            I32, # int verbosity
        ]
        self.library.minigpt4_set_verbosity.restype = None

    def panic_if_error(self, error_code: int) -> None:
        """
        Raises an exception if the error code is not 0.

        Parameters
        ----------
        error_code : int
            Error code to check.
        """

        if error_code != 0:
            raise RuntimeError(self.library.minigpt4_error_code_to_string(I32(error_code)))

    def minigpt4_model_load(self, model_path: str, llm_model_path: str, verbosity: int = 1, seed: int = 1337, n_ctx: int = 2048, n_batch: int = 512, numa: int = 0) -> MiniGPT4Context:
        """
        Loads a model from a file.

        Args:
            model_path (str): Path to model file.
            llm_model_path (str): Path to LLM model file.
            verbosity (int): Verbosity level: 0 = silent, 1 = error, 2 = info, 3 = debug. Defaults to 0.
            n_ctx (int): Size of context for llm model. Defaults to 2048.
            seed (int): Seed for llm model. Defaults to 1337.
            numa (int): NUMA node to use (0 = NUMA disabled, 1 = NUMA enabled). Defaults to 0.

        Returns:
            MiniGPT4Context: Context.
        """

        ptr = self.library.minigpt4_model_load(
            model_path.encode('utf-8'),
            llm_model_path.encode('utf-8'),
            I32(verbosity),
            I32(seed),
            I32(n_ctx),
            I32(n_batch),
            I32(numa),
        )

        assert ptr is not None, 'minigpt4_model_load failed'

        return MiniGPT4Context(ptr)

    def minigpt4_image_load_from_file(self, ctx: MiniGPT4Context, path: str, flags: int) -> MiniGPT4Image:
        """
        Loads an image from a file

        Args:
            ctx (MiniGPT4Context): context
            path (str): path
            flags (int): flags

        Returns:
            MiniGPT4Image: image
        """

        image = MiniGPT4Image()
        self.panic_if_error(self.library.minigpt4_image_load_from_file(ctx.ptr, path.encode('utf-8'), ctypes.pointer(image), I32(flags)))
        return image

    def minigpt4_preprocess_image(self, ctx: MiniGPT4Context, image: MiniGPT4Image, flags: int = 0) -> MiniGPT4Image:
        """
        Preprocesses an image

        Args:
            ctx (MiniGPT4Context): Context
            image (MiniGPT4Image): Image
            flags (int): Flags. Defaults to 0.

        Returns:
            MiniGPT4Image: Preprocessed image
        """

        preprocessed_image = MiniGPT4Image()
        self.panic_if_error(self.library.minigpt4_preprocess_image(ctx.ptr, ctypes.pointer(image), ctypes.pointer(preprocessed_image), I32(flags)))
        return preprocessed_image

    def minigpt4_encode_image(self, ctx: MiniGPT4Context, image: MiniGPT4Image, n_threads: int = 0) -> MiniGPT4Embedding:
        """
        Encodes an image into embedding

        Args:
            ctx (MiniGPT4Context): Context.
            image (MiniGPT4Image): Image.
            n_threads (int): Number of threads to use, if 0, uses all available. Defaults to 0.

        Returns:
            embedding (MiniGPT4Embedding): Output embedding.
        """

        embedding = MiniGPT4Embedding()
        self.panic_if_error(self.library.minigpt4_encode_image(ctx.ptr, ctypes.pointer(image), ctypes.pointer(embedding), n_threads))
        return embedding

    def minigpt4_begin_chat_image(self, ctx: MiniGPT4Context, image_embedding: MiniGPT4Embedding, s: str, n_threads: int = 0):
        """
        Begins a chat with an image.

        Args:
            ctx (MiniGPT4Context): Context.
            image_embedding (MiniGPT4Embedding): Image embedding.
            s (str): Question to ask about the image.
            n_threads (int, optional): Number of threads to use, if 0, uses all available. Defaults to 0.

        Returns:
            None
        """

        self.panic_if_error(self.library.minigpt4_begin_chat_image(ctx.ptr, ctypes.pointer(image_embedding), s.encode('utf-8'), n_threads))

    def minigpt4_end_chat_image(self, ctx: MiniGPT4Context, n_threads: int = 0, temp: float = 0.8, top_k: int = 40, top_p: float = 0.9, tfs_z: float = 1.0, typical_p: float = 1.0, repeat_last_n: int = 64, repeat_penalty: float = 1.1, alpha_presence: float = 1.0, alpha_frequency: float = 1.0, mirostat: int = 0, mirostat_tau: float = 5.0, mirostat_eta: float = 1.0, penalize_nl: int = 1) -> str:
        """
        Ends a chat with an image.

        Args:
            ctx (MiniGPT4Context): Context.
            n_threads (int, optional): Number of threads to use, if 0, uses all available. Defaults to 0.
            temp (float, optional): Temperature. Defaults to 0.8.
            top_k (int, optional): Top K. Defaults to 40.
            top_p (float, optional): Top P. Defaults to 0.9.
            tfs_z (float, optional): Tfs Z. Defaults to 1.0.
            typical_p (float, optional): Typical P. Defaults to 1.0.
            repeat_last_n (int, optional): Repeat last N. Defaults to 64.
            repeat_penalty (float, optional): Repeat penality. Defaults to 1.1.
            alpha_presence (float, optional): Alpha presence. Defaults to 1.0.
            alpha_frequency (float, optional): Alpha frequency. Defaults to 1.0.
            mirostat (int, optional): Mirostat. Defaults to 0.
            mirostat_tau (float, optional): Mirostat Tau. Defaults to 5.0.
            mirostat_eta (float, optional): Mirostat Eta. Defaults to 1.0.
            penalize_nl (int, optional): Penalize NL. Defaults to 1.

        Returns:
            str: Token generated.
        """

        token = CHAR_PTR()
        self.panic_if_error(self.library.minigpt4_end_chat_image(ctx.ptr, ctypes.pointer(token), n_threads, temp, top_k, top_p, tfs_z, typical_p, repeat_last_n, repeat_penalty, alpha_presence, alpha_frequency, mirostat, mirostat_tau, mirostat_eta, penalize_nl))
        return ctypes.cast(token, ctypes.c_char_p).value.decode()

    def minigpt4_system_prompt(self, ctx: MiniGPT4Context, n_threads: int = 0):
        """
        Generates a system prompt.

        Args:
            ctx (MiniGPT4Context): Context.
            n_threads (int, optional): Number of threads to use, if 0, uses all available. Defaults to 0.
        """

        self.panic_if_error(self.library.minigpt4_system_prompt(ctx.ptr, n_threads))

    def minigpt4_begin_chat(self, ctx: MiniGPT4Context, s: str, n_threads: int = 0):
        """
        Begins a chat continuing after minigpt4_begin_chat_image

        Args:
            ctx (MiniGPT4Context): Context.
            s (str): Question to ask about the image.
            n_threads (int, optional): Number of threads to use, if 0, uses all available. Defaults to 0.

        Returns:
            None
        """
        self.panic_if_error(self.library.minigpt4_begin_chat(ctx.ptr, s.encode('utf-8'), n_threads))

    def minigpt4_end_chat(self, ctx: MiniGPT4Context, n_threads: int = 0, temp: float = 0.8, top_k: int = 40, top_p: float = 0.9, tfs_z: float = 1.0, typical_p: float = 1.0, repeat_last_n: int = 64, repeat_penalty: float = 1.1, alpha_presence: float = 1.0, alpha_frequency: float = 1.0, mirostat: int = 0, mirostat_tau: float = 5.0, mirostat_eta: float = 1.0, penalize_nl: int = 1) -> str:
        """
        Ends a chat.

        Args:
            ctx (MiniGPT4Context): Context.
            n_threads (int, optional): Number of threads to use, if 0, uses all available. Defaults to 0.
            temp (float, optional): Temperature. Defaults to 0.8.
            top_k (int, optional): Top K. Defaults to 40.
            top_p (float, optional): Top P. Defaults to 0.9.
            tfs_z (float, optional): Tfs Z. Defaults to 1.0.
            typical_p (float, optional): Typical P. Defaults to 1.0.
            repeat_last_n (int, optional): Repeat last N. Defaults to 64.
            repeat_penalty (float, optional): Repeat penality. Defaults to 1.1.
            alpha_presence (float, optional): Alpha presence. Defaults to 1.0.
            alpha_frequency (float, optional): Alpha frequency. Defaults to 1.0.
            mirostat (int, optional): Mirostat. Defaults to 0.
            mirostat_tau (float, optional): Mirostat Tau. Defaults to 5.0.
            mirostat_eta (float, optional): Mirostat Eta. Defaults to 1.0.
            penalize_nl (int, optional): Penalize NL. Defaults to 1.

        Returns:
            str: Token generated.
        """

        token = CHAR_PTR()
        self.panic_if_error(self.library.minigpt4_end_chat(ctx.ptr, ctypes.pointer(token), n_threads, temp, top_k, top_p, tfs_z, typical_p, repeat_last_n, repeat_penalty, alpha_presence, alpha_frequency, mirostat, mirostat_tau, mirostat_eta, penalize_nl))
        return ctypes.cast(token, ctypes.c_char_p).value.decode()

    def minigpt4_reset_chat(self, ctx: MiniGPT4Context):
        """
        Resets the chat.

        Args:
            ctx (MiniGPT4Context): Context.
        """
        self.panic_if_error(self.library.minigpt4_reset_chat(ctx.ptr))

    def minigpt4_contains_eos_token(self, s: str) -> bool:

        """
        Checks if a string contains an EOS token.

        Args:
            s (str): String to check.
        
        Returns:
            bool: True if the string contains an EOS token, False otherwise.
        """

        return self.library.minigpt4_contains_eos_token(s.encode('utf-8'))

    def minigpt4_is_eos(self, s: str) -> bool:

        """
        Checks if a string is EOS.

        Args:
            s (str): String to check.
        
        Returns:
            bool: True if the string contains an EOS, False otherwise.
        """

        return self.library.minigpt4_is_eos(s.encode('utf-8'))


    def minigpt4_free(self, ctx: MiniGPT4Context) -> None:
        """
        Frees a context.

        Args:
            ctx (MiniGPT4Context): Context.
        """

        self.panic_if_error(self.library.minigpt4_free(ctx.ptr))

    def minigpt4_free_image(self, image: MiniGPT4Image) -> None:
        """
        Frees an image.

        Args:
            image (MiniGPT4Image): Image.
        """

        self.panic_if_error(self.library.minigpt4_free_image(ctypes.pointer(image)))

    def minigpt4_free_embedding(self, embedding: MiniGPT4Embedding) -> None:
        """
        Frees an embedding.

        Args:
            embedding (MiniGPT4Embedding): Embedding.
        """

        self.panic_if_error(self.library.minigpt4_free_embedding(ctypes.pointer(embedding)))

    def minigpt4_error_code_to_string(self, error_code: int) -> str:
        """
        Converts an error code to a string.

        Args:
            error_code (int): Error code.

        Returns:
            str: Error string.
        """

        return self.library.minigpt4_error_code_to_string(error_code).decode()

    def minigpt4_quantize_model(self, in_path: str, out_path: str, data_type: DataType):
        """
        Quantizes a model file.

        Args:
            in_path (str): Path to input model file.
            out_path (str): Path to write output model file.
            data_type (DataType): Must be one DataType enum values.
        """

        self.panic_if_error(self.library.minigpt4_quantize_model(in_path.encode('utf-8'), out_path.encode('utf-8'), data_type))

    def minigpt4_set_verbosity(self, verbosity: Verbosity):
        """
        Sets verbosity.

        Args:
            verbosity (int): Verbosity.
        """

        self.library.minigpt4_set_verbosity(I32(verbosity))

def load_library() -> MiniGPT4SharedLibrary:
    """
    Attempts to find minigpt4.cpp shared library and load it.
    """

    file_name: str

    if 'win32' in sys.platform or 'cygwin' in sys.platform:
        file_name = 'minigpt4.dll'
    elif 'darwin' in sys.platform:
        file_name = 'libminigpt4.dylib'
    else:
        file_name = 'libminigpt4.so'

    cwd = pathlib.Path(os.getcwd())
    repo_root_dir: pathlib.Path = pathlib.Path(os.path.abspath(__file__)).parent.parent

    paths = [
        # If we are in "minigpt4" directory
        f'../bin/Release/{file_name}',
        # If we are in repo root directory
        f'bin/Release/{file_name}',
        # If we compiled in build directory
        f'build/bin/Release/{file_name}',
        # If we compiled in build directory
        f'build/{file_name}',
        f'../build/{file_name}',
        # Search relative to this file
        str(repo_root_dir / 'bin' / 'Release' / file_name),
        str(repo_root_dir / 'build' / 'bin' / 'Release' / file_name),
        str(repo_root_dir / 'bin' / 'Debug' / file_name),
        str(repo_root_dir / 'build' / 'bin' / 'Debug' / file_name),
        # Fallback
        str(repo_root_dir / file_name),
        str(cwd / file_name)
    ]

    for path in paths:
        if os.path.isfile(path):
            return MiniGPT4SharedLibrary(path)

    return MiniGPT4SharedLibrary(paths[-1])

class MiniGPT4ChatBot:
    def __init__(self, model_path: str, llm_model_path: str, verbosity: Verbosity = Verbosity.SILENT, n_threads: int = 0):
        """
        Creates a new MiniGPT4ChatBot instance.

        Args:
            model_path (str): Path to model file.
            llm_model_path (str): Path to language model model file.
            verbosity (Verbosity, optional): Verbosity. Defaults to Verbosity.SILENT.
            n_threads (int, optional): Number of threads to use. Defaults to 0.
        """
            
        self.library = load_library()
        self.ctx = self.library.minigpt4_model_load(model_path, llm_model_path, verbosity)
        self.n_threads = n_threads

        from PIL import Image
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode
        self.image_size = 224

        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.image_size,
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        )
        self.embedding: Optional[MiniGPT4Embedding] = None
        self.is_image_chat = False
        self.chat_history = []

    def free(self):
        if self.ctx:
            self.library.minigpt4_free(self.ctx)

    def generate(self, message: str, limit: int = 1024, temp: float = 0.8, top_k: int = 40, top_p: float = 0.9, tfs_z: float = 1.0, typical_p: float = 1.0, repeat_last_n: int = 64, repeat_penalty: float = 1.1, alpha_presence: float = 1.0, alpha_frequency: float = 1.0, mirostat: int = 0, mirostat_tau: float = 5.0, mirostat_eta: float = 1.0, penalize_nl: int = 1):
        """
        Generates a chat response.

        Args:
            message (str): Message.
            limit (int, optional): Limit. Defaults to 1024.
            temp (float, optional): Temperature. Defaults to 0.8.
            top_k (int, optional): Top K. Defaults to 40.
            top_p (float, optional): Top P. Defaults to 0.9.
            tfs_z (float, optional): TFS Z. Defaults to 1.0.
            typical_p (float, optional): Typical P. Defaults to 1.0.
            repeat_last_n (int, optional): Repeat last N. Defaults to 64.
            repeat_penalty (float, optional): Repeat penalty. Defaults to 1.1.
            alpha_presence (float, optional): Alpha presence. Defaults to 1.0.
            alpha_frequency (float, optional): Alpha frequency. Defaults to 1.0.
            mirostat (int, optional): Mirostat. Defaults to 0.
            mirostat_tau (float, optional): Mirostat tau. Defaults to 5.0.
            mirostat_eta (float, optional): Mirostat eta. Defaults to 1.0.
            penalize_nl (int, optional): Penalize NL. Defaults to 1.
        """
        if self.is_image_chat:
            self.is_image_chat = False
            self.library.minigpt4_begin_chat_image(self.ctx, self.embedding, message, self.n_threads)
            chat = ''
            for _ in range(limit):
                token = self.library.minigpt4_end_chat_image(self.ctx, self.n_threads, temp, top_k, top_p, tfs_z, typical_p, repeat_last_n, repeat_penalty, alpha_presence, alpha_frequency, mirostat, mirostat_tau, mirostat_eta, penalize_nl)
                chat += token
                if self.library.minigpt4_contains_eos_token(token):
                    continue
                if self.library.minigpt4_is_eos(chat):
                    break
                yield token
        else:
            self.library.minigpt4_begin_chat(self.ctx, message, self.n_threads)
            chat = ''
            for _ in range(limit):
                try:
                    token = self.library.minigpt4_end_chat(self.ctx, self.n_threads, temp, top_k, top_p, tfs_z, typical_p, repeat_last_n, repeat_penalty, alpha_presence, alpha_frequency, mirostat, mirostat_tau, mirostat_eta, penalize_nl)
                except KeyboardInterrupt:
                    break
                except Exception as exception:
                    raise exception
                chat += token
                if self.library.minigpt4_contains_eos_token(token):
                    continue
                if self.library.minigpt4_is_eos(chat):
                    break
                yield token

    def reset_chat(self):
        """
        Resets the chat.
        """

        self.is_image_chat = False
        if self.embedding:
            self.library.minigpt4_free_embedding(self.embedding)
            self.embedding = None

        self.library.minigpt4_reset_chat(self.ctx)
        self.library.minigpt4_system_prompt(self.ctx, self.n_threads)

    def upload_image(self, image):
        """
        Uploads an image.
        
        Args:
            image (Image): Image.
        """

        self.reset_chat()

        image = self.transform(image)
        image = image.unsqueeze(0)
        image = image.numpy()
        image = image.ctypes.data_as(ctypes.c_void_p)
        minigpt4_image = MiniGPT4Image(image, self.image_size, self.image_size, 3, ImageFormat.F32)
        self.embedding = self.library.minigpt4_encode_image(self.ctx, minigpt4_image, self.n_threads)
        
        self.is_image_chat = True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test loading minigpt4')
    parser.add_argument('model_path', help='Path to model file')
    parser.add_argument('llm_model_path', help='Path to llm model file')
    parser.add_argument('-i', '--image_path', help='Image to test', default='images/llama.png')
    parser.add_argument('-p', '--prompts', help='Text to test', default='what is the text in the picture?,what is the color of it?')
    parser.add_argument('-t', '--test_native_image_implementation', help='Test native image code', default=False)
    args = parser.parse_args()

    model_path = args.model_path
    llm_model_path = args.llm_model_path
    image_path = args.image_path
    prompts = args.prompts
    test_native_image_implementation = args.test_native_image_implementation

    if not Path(model_path).exists():
        print(f'Model does not exist: {model_path}')
        exit(1) 

    if not Path(llm_model_path).exists():
        print(f'LLM Model does not exist: {llm_model_path}')
        exit(1)

    prompts = prompts.split(',')

    print('Loading minigpt4 shared library...')
    library = load_library()
    print(f'Loaded library {library}')
    ctx = library.minigpt4_model_load(model_path, llm_model_path, Verbosity.DEBUG)
    if test_native_image_implementation:
        image = library.minigpt4_image_load_from_file(ctx, image_path, 0)
        preprocessed_image = library.minigpt4_preprocess_image(ctx, image, 0)
    else:
        from PIL import Image
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode
        image_size = 224

        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        )
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0)
        image = image.numpy()
        image = image.ctypes.data_as(ctypes.c_void_p)
        preprocessed_image = MiniGPT4Image(image, image_size, image_size, 3, ImageFormat.F32)

    question = prompts[0]
    n_threads = 0
    embedding = library.minigpt4_encode_image(ctx, preprocessed_image, n_threads)
    library.minigpt4_system_prompt(ctx, n_threads)
    library.minigpt4_begin_chat_image(ctx, embedding, question, n_threads)
    chat = ''
    while True:
        token = library.minigpt4_end_chat_image(ctx, n_threads)
        chat += token
        if library.minigpt4_contains_eos_token(token):
            continue
        if library.minigpt4_is_eos(chat):
            break
        print(token, end='', flush=True)

    for i in range(1, len(prompts)):
        prompt = prompts[i]
        library.minigpt4_begin_chat(ctx, prompt, n_threads)
        chat  = ''
        while True:
            try:
                token = library.minigpt4_end_chat(ctx, n_threads)
            except KeyboardInterrupt:
                break
            except Exception as exception:
                raise exception
            chat += token
            if library.minigpt4_contains_eos_token(token):
                continue
            if library.minigpt4_is_eos(chat):
                break
            print(token, end='', flush=True)

    if test_native_image_implementation:
        library.minigpt4_free_image(image)
        library.minigpt4_free_image(preprocessed_image)
    library.minigpt4_free(ctx)
