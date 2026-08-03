; AeroVision CFD - Inno Setup installer betigi
; Kullanim (Windows):
;   1) once onedir paketi uretilir:  python -m PyInstaller build.spec --clean
;   2) Inno Setup Compiler ile bu betik derlenir (Setup Compiler > Build)
; Cikti: dist\AeroVision_CFD_Setup.exe
; Not: MyAppVersion'i pyproject.toml ile ayni tutun (release versiyonu).

#define MyAppName "AeroVision CFD"
#define MyAppVersion "1.1.0"
#define MyAppPublisher "AeroVision"
#define MyAppExeName "AeroVision_CFD.exe"
#define MyAppURL "https://github.com/Xeakaes/AeroVision-CFD"

[Setup]
AppId={{1B0C38B3-69C8-4E44-B9F3-F1776371E779}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=LICENSE
OutputDir=dist
OutputBaseFilename=AeroVision_CFD_Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "turkish"; MessagesFile: "compiler:Languages\Turkish.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "dist\AeroVision_CFD\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
