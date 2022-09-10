# MIT License
#
# Copyright (c) 2022 Resha Dwika Hefni Al-Fahsi
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================


from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File

from imagecaptioning.__about__ import __description__ as DESCRIPTION
from imagecaptioning.model import ImageCaptioning


app = FastAPI()
image_captioning = ImageCaptioning()


"""Sanity Check"""


@app.get("/")
async def index():
    return DESCRIPTION


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    return image_captioning.predict(file)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
