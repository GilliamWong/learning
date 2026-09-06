"""Package the plain-JavaScript local extension using the standard VSIX layout."""

import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "vscode-extension"
RUNTIME = ROOT / ".runtime"
OUTPUT = RUNTIME / "mech-interp-workbench.vsix"


def package():
    files = sorted(p for p in SOURCE.rglob("*") if p.is_file() and p.suffix in {".js", ".json", ".css", ".svg", ".md"})
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(SOURCE).as_posix().encode())
        digest.update(path.read_bytes())
    fingerprint = digest.hexdigest()
    marker = RUNTIME / "vscode-extension-source.sha256"
    RUNTIME.mkdir(exist_ok=True)
    if OUTPUT.exists() and marker.exists() and marker.read_text() == fingerprint:
        return OUTPUT
    project = json.loads((SOURCE / "package.json").read_text(encoding="utf-8"))
    manifest = f'''<?xml version="1.0" encoding="utf-8"?>
<PackageManifest Version="2.0.0" xmlns="http://schemas.microsoft.com/developer/vsx-schema/2011">
  <Metadata><Identity Language="en-US" Id="{project['name']}" Version="{project['version']}" Publisher="{project['publisher']}"/>
    <DisplayName>{escape(project['displayName'])}</DisplayName><Description xml:space="preserve">{escape(project['description'])}</Description>
    <Categories>Other,Notebooks</Categories><Properties>
      <Property Id="Microsoft.VisualStudio.Code.Engine" Value="{project['engines']['vscode']}"/>
      <Property Id="Microsoft.VisualStudio.Code.ExtensionDependencies" Value="{','.join(project['extensionDependencies'])}"/>
      <Property Id="Microsoft.VisualStudio.Code.ExtensionKind" Value="workspace"/>
      <Property Id="Microsoft.VisualStudio.Code.ExecutesCode" Value="true"/>
    </Properties></Metadata>
  <Installation><InstallationTarget Id="Microsoft.VisualStudio.Code"/></Installation><Dependencies/>
  <Assets><Asset Type="Microsoft.VisualStudio.Code.Manifest" Path="extension/package.json" Addressable="true"/>
    <Asset Type="Microsoft.VisualStudio.Services.Content.Details" Path="extension/README.md" Addressable="true"/></Assets>
</PackageManifest>'''
    content_types = '''<?xml version="1.0" encoding="utf-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="json" ContentType="application/json"/><Default Extension="js" ContentType="application/javascript"/>
<Default Extension="css" ContentType="text/css"/><Default Extension="svg" ContentType="image/svg+xml"/>
<Default Extension="md" ContentType="text/markdown"/><Default Extension="vsixmanifest" ContentType="text/xml"/>
</Types>'''
    with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as archive:
        archive.writestr("extension.vsixmanifest", manifest)
        archive.writestr("[Content_Types].xml", content_types)
        for path in files:
            archive.write(path, "extension/" + path.relative_to(SOURCE).as_posix())
    marker.write_text(fingerprint)
    return OUTPUT


if __name__ == "__main__":
    print(package())
