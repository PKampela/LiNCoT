#define MyAppName "LiNCoT"
#define MyAppVersion "0.5.0"
#define MyAppPublisher "LiNCoT"
#define MyAppExeName "LiNCoT.exe"

[Setup]
AppId={{709311BB-5B51-4C32-8100-8680659A4935}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}

DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}

OutputDir=..\dist-installer
OutputBaseFilename=LiNCoT-{#MyAppVersion}-Setup

Compression=lzma
SolidCompression=yes

WizardStyle=modern

UninstallDisplayIcon={app}\{#MyAppExeName}

[Files]
Source: "..\dist\LiNCoT\*"; DestDir: "{app}"; \
    Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\LiNCoT"; Filename: "{app}\{#MyAppExeName}"

Name: "{autodesktop}\LiNCoT"; Filename: "{app}\{#MyAppExeName}"

[Run]
Filename: "{app}\{#MyAppExeName}"; \
    Description: "Launch LiNCoT"; \
    Flags: nowait postinstall skipifsilent